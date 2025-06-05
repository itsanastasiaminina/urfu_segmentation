import torch
import torch.nn as nn
import torch.nn.functional as F

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