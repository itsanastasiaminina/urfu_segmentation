import torch
import torch.nn as nn
import torch.nn.functional as F

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
        