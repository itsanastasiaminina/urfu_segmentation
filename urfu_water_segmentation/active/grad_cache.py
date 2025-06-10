import torch
import torch.nn as nn

class GradCache:
    def __init__(self, beta: float = .9):
        self.beta = beta
        self.buf: torch.Tensor | None = None

    def update(self, g: torch.Tensor) -> torch.Tensor:
        if self.buf is None:
            self.buf = g.clone()
        else:
            self.buf = self.beta * self.buf + (1 - self.beta) * g
        return self.buf.clone()

    @property
    def value(self) -> torch.Tensor:
        if self.buf is None:
            raise RuntimeError("Grad-cache empty – call update() first")
        return self.buf.clone()