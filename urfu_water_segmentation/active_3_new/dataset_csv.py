import csv, mmcv, torch
import numpy as np
from mmseg.registry import DATASETS
from dataset import LandcoverAI


@DATASETS.register_module()
class LandcoverAI_CSV(LandcoverAI):

    def load_data_list(self):
        if not self.ann_file:
            raise ValueError('`ann_file` для LandcoverAI_CSV не задан')

        data_list = []
        with open(self.ann_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                info = dict(
                    img_path=row['img_path'],
                    seg_map_path=row['gt_path'],
                    reduce_zero_label=self.reduce_zero_label,
                    seg_fields=[]              
                )
                data_list.append(info)
        return data_list
    
    @staticmethod
    def load_img(path: str) -> torch.Tensor:
        img = mmcv.imread(path, flag='color')[..., ::-1].copy()
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    
    @staticmethod
    def load_mask(mask_path: str) -> torch.Tensor:
        m = mmcv.imread(mask_path, flag='grayscale')
        return torch.from_numpy((m == 64).astype(np.uint8)).long()