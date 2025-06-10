from evaluator import QuickEvaluator
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging
from typing import List, Dict
from pathlib import Path
import cv2
import numpy as np
from dataset_csv import LandcoverAI_CSV

BATCH_SIZE = 8
WATER_PIXEL_VALUE = 64
BACKGROUND_IDX = 0
WATER_IDX = 1

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