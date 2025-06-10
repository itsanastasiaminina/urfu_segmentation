from typing import Iterable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def _flatten(params: Iterable[nn.Parameter]) -> torch.Tensor:
    parts = [p.grad.detach().cpu().flatten()
             for p in params if p.grad is not None]
    return torch.cat(parts) if parts else torch.zeros(0)

def _run(
    model: nn.Module,
    loss_fn: nn.Module,
    imgs: torch.Tensor,
    masks: torch.Tensor,
    device: str = 'cpu',
    retain_graph: bool = False
) -> Tuple[torch.Tensor, float]:

    model = model.to(device).eval()
    imgs  = imgs.to(device)
    masks = masks.to(device)

    with torch.enable_grad():
        feats  = model.extract_feat(imgs)         
        logits = model.decode_head(feats)          

        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits, size=masks.shape[-2:], mode='bilinear',
                align_corners=False
            )

        loss = loss_fn(logits, masks)

        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=retain_graph)
        g_vec = _flatten(model.parameters())

    return g_vec, float(loss.item())

def single_grad(
    model: nn.Module,
    loss_fn: nn.Module,
    sample: Tuple[torch.Tensor, torch.Tensor],
    *,
    device: str = 'cpu',
    retain_graph: bool = False
):
    img, mask = sample
    return _run(model, loss_fn,
                img.unsqueeze(0), mask.unsqueeze(0),
                device, retain_graph)


def batch_grad(
    model: nn.Module,
    loss_fn: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    *,
    device: str = 'cpu',
    retain_graph: bool = False
):
    imgs, masks = batch
    return _run(model, loss_fn, imgs, masks, device, retain_graph)
