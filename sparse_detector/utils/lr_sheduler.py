import warnings

from torch.optim import lr_scheduler
from torch.optim import Optimizer


class StepLRExcludeAlpha(lr_scheduler.StepLR):
    """Extend Pytorch's StepLR. Decays the learning rate of each parameter group
    by gamma every step_size epochs, except for some param groups
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False,
        exclude_alpha: bool = True,
    ) -> None:
        self.exclude_alpha = exclude_alpha
        super().__init__(optimizer, step_size, gamma, last_epoch, verbose)

    def get_lr(self) -> float:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group["lr"] for group in self.optimizer.param_groups]

        if self.exclude_alpha:
            learning_rates = []
            for group in self.optimizer.param_groups:
                if len(group["params"]) == 6:  # `alpha` parameters group
                    learning_rates.append[group["lr"]]
                else:
                    learning_rates.append[group["lr"] * self.gamma]

            return learning_rates

        # Normal case
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [
            base_lr * self.gamma ** (self.last_epoch // self.step_size)
            for base_lr in self.base_lrs
        ]
