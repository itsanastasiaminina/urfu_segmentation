import argparse
import logging
import random
import math
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from mmengine.config import Config
from mmengine.runner import Runner

from mmseg.apis import MMSegInferencer
from mmseg.utils import register_all_modules
from mmseg.models.losses import FocalLoss

from dataset_csv import LandcoverAI_CSV
from grad_cache import GradCache
from grad_utils import batch_grad

import torch
import torch.nn as nn
import torch.nn.functional as F

_DATA_ROOT = Path('/misc/home1/m_imm_freedata/Segmentation')
DATASETS: List[Dict[str, Path]] = [
    {
        'name': 'landcover',
        'images': _DATA_ROOT / 'Projects/mmseg_water/landcover.ai_512/val/images',
        'gt': _DATA_ROOT / 'Projects/mmseg_water/landcover.ai_512/val/gt',
    },
    {
        'name': 'glh_water',
        'images': _DATA_ROOT / 'Projects/GLH_water/glh_cut_512_filtered/val/images',
        'gt': _DATA_ROOT / 'Projects/GLH_water/glh_cut_512_filtered/val/gt',
    },
    {
        'name': 'deepglobe',
        'images': _DATA_ROOT / 'DeepGlobe_Land/DeepGlobe512/val/images',
        'gt': _DATA_ROOT / 'DeepGlobe_Land/DeepGlobe512/val/gt',
    },
    {
        'name': 'loveda',
        'images': _DATA_ROOT / 'LoveDA/val/images',
        'gt': _DATA_ROOT / 'LoveDA/val/gt',
    },
    {
        'name': 'rg3',
        'images': _DATA_ROOT / 'RG3/train_and_val/images',
        'gt': _DATA_ROOT / 'RG3/train_and_val/gt',
    },
]

BATCH_SIZE = 8
WATER_PIXEL_VALUE = 64
BACKGROUND_IDX = 0
WATER_IDX = 1

DATA_DIR = Path("data")
LOG_DIR = Path("logs")

TAU = 0.0
# DEFAULT_AL_BATCH = 20
TEST_FRAC = 0.20
GRAD_BS = 8

IF_PREPARED = 500
PREPARED = 2000

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s")


def _csv_path(kind: str, k: int) -> Path:
    return DATA_DIR / f"{kind}_{k}.csv"


def _save_csv(df: pd.DataFrame, kind: str, k: int) -> None:
    p = _csv_path(kind, k)
    p.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(p, index=False)


def _latest_pth(work_dir: Path) -> Path:
    ptr = work_dir / 'last_checkpoint'
    if ptr.is_file():
        with open(ptr) as fp:
            ckpt = Path(fp.readlines()[-1].strip())
        if ckpt.is_file():
            return ckpt
    return work_dir / 'latest.pth'


def to_tensors(batch_dict: dict, device: str = 'cpu'):
    imgs = torch.stack([LandcoverAI_CSV.load_img(p) for p in batch_dict['img_path']]).to(device)
    masks = torch.stack([LandcoverAI_CSV.load_mask(p) for p in batch_dict['seg_map_path']]).to(device)
    return imgs, masks


class QuickPredictor:
    def __init__(self, model: torch.nn.Module, half: bool = True):
        self.model = model.eval()
        self.half = half

    @torch.no_grad()
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        ctx = torch.cuda.amp.autocast() if self.half else torch.no_grad()
        with ctx:
            feats = self.model.extract_feat(img.unsqueeze(0))
            logits = self.model.decode_head(feats)
            if logits.shape[-2:] != img.shape[-2:]:
                logits = torch.nn.functional.interpolate(logits, img.shape[-2:], mode='bilinear', align_corners=False)
            return logits.argmax(1)[0].byte()

class QuickEvaluator:
    def __init__(self, model: torch.nn.Module, half: bool = True):
        self.model = model.eval()
        self.half = half

    @torch.no_grad()
    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:  
        ctx = torch.cuda.amp.autocast() if self.half else torch.no_grad()
        with ctx:
            feats = self.model.extract_feat(imgs)
            logits = self.model.decode_head(feats)
            if logits.shape[-2:] != imgs.shape[-2:]:
                logits = torch.nn.functional.interpolate(logits, imgs.shape[-2:], mode='bilinear', align_corners=False)
            return logits.argmax(1).byte() 
        
def evaluate_water(model: torch.nn.Module, device: str) -> Dict[str, Dict[str, float]]:
    predictor = QuickEvaluator(model, half=True)
    results: Dict[str, Dict[str, float]] = {}

    for ds in DATASETS:
        img_dir, gt_dir = ds['images'], ds['gt']
        img_paths = sorted(img_dir.glob('*.tif'))
        logging.info(f"ds {ds}")
        gt_maps: List[torch.Tensor] = []
        logging.info("gt_maps")
        for gt_path in sorted(gt_dir.glob('*.tif')):
            gt_np = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            gt_t  = torch.from_numpy(gt_np).to(device)
            gt_maps.append((gt_t == WATER_PIXEL_VALUE))  

        pred_maps: List[torch.Tensor] = []
        logging.info("pred_maps")
        for i in range(0, len(img_paths), BATCH_SIZE):
            batch_paths = img_paths[i:i + BATCH_SIZE]
            imgs = torch.stack([LandcoverAI_CSV.load_img(str(p)) for p in batch_paths]).to(device)
            batch_pred = predictor(imgs)  
            pred_maps.extend(batch_pred.bool())
            del imgs, batch_pred
            torch.cuda.empty_cache()

        iou_vals, acc_vals = [], []
        for pred, gt in zip(pred_maps, gt_maps):
            intersect = (pred & gt).sum().item()
            union = (pred | gt).sum().item()
            iou_vals.append(intersect / (union + 1e-8))
            acc_vals.append(intersect / (gt.sum().item() + 1e-8))

        results[ds['name']] = {
            'mAcc_water': float(np.mean(acc_vals)),
            'mIoU_water': float(np.mean(iou_vals)),
        }
        logging.info("Eval %s: mAcc=%.4f mIoU=%.4f", ds['name'],
                     results[ds['name']]['mAcc_water'], results[ds['name']]['mIoU_water'])

    return results

def train_once(cfg_path: str, k: int, device: str):
    cfg = Config.fromfile(cfg_path)
    tr_ds = cfg.train_dataloader.dataset
    tr_ds.type = 'LandcoverAI_CSV'
    tr_ds.data_root = ''
    tr_ds.ann_file = str(_csv_path('labeled', k).resolve())
    tr_ds.data_prefix = dict(img_path='', seg_map_path='')
    for split in ('val_dataloader', 'test_dataloader'):
        ds = getattr(cfg, split).dataset
        ds.type = 'LandcoverAI'
        ds.pop('ann_file', None)
    cfg.work_dir = str(LOG_DIR / f'iter_{k}')
    # for key in ("default_hooks", "visualizer", "vis_backends"):
    #     cfg.pop(key, None)

    if k > 0:
        cfg.load_from = str(_latest_pth(LOG_DIR / f'iter_{k-1}'))
    runner = Runner.from_cfg(cfg)
    runner.train()
    return runner.model

def active_loop(max_iter: int, cfg_path: str, device: str):
    register_all_modules()

    for k in range(max_iter):
        logging.info('========== AL‑цикл %d ==========', k)

        labeled_df = pd.read_csv(_csv_path('labeled', k))
        unlabeled_df = pd.read_csv(_csv_path('unlabeled', k))
        annot_df = pd.read_csv(_csv_path('annotator', k))
        DEFAULT_AL_BATCH = len(unlabeled_df) * 0.2

        unlabeled_count = len(unlabeled_df)
        al_batch = IF_PREPARED if unlabeled_count < PREPARED else DEFAULT_AL_BATCH
        logging.info('  → AL_BATCH=%d (unlabeled=%d)', al_batch, unlabeled_count)

        if unlabeled_count < DEFAULT_AL_BATCH:
            logging.info('  → Осталось <%d неразмеченных. Забираем все и останавливаемся.', DEFAULT_AL_BATCH)
            new_labeled_df = annot_df.copy().reset_index(drop=True)
            _save_csv(new_labeled_df, 'labeled', k + 1)
            _save_csv(pd.DataFrame(columns=unlabeled_df.columns), 'unlabeled', k + 1)
            _save_csv(pd.DataFrame(columns=annot_df.columns), 'annotator', k + 1)
            break

        base_model = train_once(cfg_path, k, device).to(device)
        base_model.eval()
        torch.cuda.empty_cache()

        iter_dir = LOG_DIR / f'iter_{k}'
        logging.info("start evaluation")

        metrics = evaluate_water(base_model, device)
        with open(iter_dir / 'water_metrics.txt', 'w') as fp:
            for ds_name, vals in metrics.items():
                fp.write(f"{ds_name} mAcc_water={vals['mAcc_water']:.4f} mIoU_water={vals['mIoU_water']:.4f}\n")

        loss_fn = torch.nn.CrossEntropyLoss()
        loss_fn = FocalLoss(class_weight=[0.9, 1.1], gamma=2)
        loss_fn = FocalLoss(class_weight=[0.9, 1.1]) 


        qpredict = QuickPredictor(base_model, half=True)

        train_ds = LandcoverAI_CSV(
            ann_file=str(_csv_path("labeled", k)),
            data_root="",
            data_prefix=dict(img_path="", seg_map_path=""),
            pipeline=[],
        )

        bs_grad = min(GRAD_BS, len(train_ds))
        grad_loader = DataLoader(train_ds, batch_size=bs_grad, shuffle=True, num_workers=4, pin_memory=True)

        te_idx = random.sample(range(len(train_ds)), max(1, int(len(train_ds) * TEST_FRAC)))
        test_loader = DataLoader(Subset(train_ds, te_idx), batch_size=bs_grad, shuffle=False, num_workers=2, pin_memory=True)

        base_imgs, base_masks = to_tensors(next(iter(grad_loader)), device)
        test_imgs, test_masks = to_tensors(next(iter(test_loader)), device)

        with torch.cuda.amp.autocast():
            g_base, _ = batch_grad(base_model, loss_fn, (base_imgs, base_masks), device=device)
            g_test, _ = batch_grad(base_model, loss_fn, (test_imgs, test_masks), device=device)

        cache = GradCache(); cache.update(g_test)

        selected_idx: List[int] = []
        phi_log: List[tuple[int, str, float]] = []

        for global_i, row in unlabeled_df.iterrows():
            img_path = row["img_path"]
            g_img = LandcoverAI_CSV.load_img(img_path).to(device)
            raw_pred = qpredict(g_img)
            if raw_pred is None or raw_pred.numel() == 0:
                logging.error(f"No prediction generated for image at {img_path}")
                continue

            g_mask = raw_pred.long().to(device)
            withg_batch = (
                torch.cat([base_imgs, g_img.unsqueeze(0)], 0),
                torch.cat([base_masks, g_mask.unsqueeze(0)], 0),
            )

            with torch.cuda.amp.autocast():
                g_full, _ = batch_grad(base_model, loss_fn, withg_batch, device=device)
            phi = torch.dot(g_base - g_full, cache.value).item()
            phi_log.append((global_i, img_path, phi))

            del g_full, withg_batch, g_mask, g_img, raw_pred
            torch.cuda.empty_cache()
            if phi > TAU:
                logging.info(f"global_i={global_i} phi={phi} selected")
                selected_idx.append(global_i)
            else:
                logging.info(f"global_i={global_i} phi={phi}")
            if len(selected_idx) >= al_batch:
                break

        if len(selected_idx) < al_batch:
            phi_sorted = sorted(phi_log, key=lambda x: x[2], reverse=True)
            for idx, _, _ in phi_sorted:
                if idx not in selected_idx:
                    selected_idx.append(idx)
                if len(selected_idx) >= al_batch:
                    break

        phi_df = pd.DataFrame(phi_log, columns=["df_index", "img_path", "phi"])
        iter_dir = LOG_DIR / f"iter_{k}"; iter_dir.mkdir(parents=True, exist_ok=True)
        phi_df.to_csv(iter_dir / "phi_scores.csv", index=False)

        new_labeled_df = annot_df.loc[selected_idx].reset_index(drop=True)
        _save_csv(new_labeled_df, "labeled", k + 1)
        _save_csv(unlabeled_df.drop(index=selected_idx), "unlabeled", k + 1)
        _save_csv(annot_df.drop(index=selected_idx), "annotator", k + 1)

        logging.info("Цикл %d завершён: +%d образцов", k, len(selected_idx))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="active_3/config_landcover.py")
    parser.add_argument("--max_iter", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    active_loop(args.max_iter, args.cfg, args.device)
