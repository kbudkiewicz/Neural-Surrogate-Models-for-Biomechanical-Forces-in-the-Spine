from . import *
import torch.nn.functional as F


class ScaledRMSELoss(nn.MSELoss):
    def __init__(self, reduction):
        super().__init__(reduction=reduction)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.sqrt(F.mse_loss(x, y, reduction=self.reduction))


class ScaledLoss:
    """Scale a given loss by standard deviation across each batch element."""
    def __init__(self, criterion):
        if not issubclass(criterion, nn.modules.loss._Loss):
            raise TypeError(f'criterion must be a subclass of _Loss')
        self.criterion = criterion(reduction='none')

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = self.criterion.forward(pred, target)
        loss = loss / (loss.std(dim=0) + 1e-6)      # per batch
        # loss = loss.T / (loss.std(dim=1) + 1e-6)    # per target
        return loss.mean()
