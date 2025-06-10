import torch
import torch.nn as nn
from grad_utils import batch_grad, single_grad
from grad_cache import GradCache

def contribution(
    model: nn.Module,
    loss_fn: nn.Module,
    batch_base,       
    batch_with_g,     
    test_batch,        
    cache: GradCache,
    alpha: float = 1.,
) -> float:
    g_base, _  = batch_grad(model, loss_fn, batch_base)
    g_full, _  = batch_grad(model, loss_fn, batch_with_g)
    g_test, _  = batch_grad(model, loss_fn, test_batch)
    ḡ_u        = cache.update(g_test)
    return alpha * torch.dot(g_base - g_full, ḡ_u).item()