import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, train_config, model_config, current_step):

        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=train_config["optimizer"]["betas"],
            eps=train_config["optimizer"]["eps"],
            weight_decay=train_config["optimizer"]["weight_decay"],
        )
        self.n_warmup_steps = train_config["optimizer"]["warm_up_step"]
        self.anneal_steps = train_config["optimizer"]["anneal_steps"]
        self.anneal_rate = train_config["optimizer"]["anneal_rate"]
        self.current_step = current_step
        self.init_lr = np.power(model_config["transformer"]["encoder_hidden"], -0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
            # param_group['lr'] = .0001

    def get_lr(self):
        return self._optimizer.param_groups[0]["lr"]

    @property
    def param_groups(self):
        return self._optimizer.param_groups


class CyclicDecayLR(_LRScheduler):
        def __init__(self, optimizer, config, last_epoch=-1):
            self.A = config['cyclic_optimizer']['A']
            self.gamma = config['cyclic_optimizer']['gamma']
            self.freq = config['cyclic_optimizer']['freq']
            self.lambd = config['cyclic_optimizer']['lambd']
            self.max_lr = config['cyclic_optimizer']['max_lr']
            self.min_lr = config['cyclic_optimizer']['min_lr']
            self.last_epoch = last_epoch
            super(CyclicDecayLR, self).__init__(optimizer, last_epoch)

        def get_lr(self):
            lr = self.A * np.exp(-self.gamma * self.last_epoch) * np.sin(self.last_epoch * self.freq) + \
                np.exp(-self.lambd * self.last_epoch) * self.max_lr + self.min_lr
            return [lr for _ in self.optimizer.param_groups]