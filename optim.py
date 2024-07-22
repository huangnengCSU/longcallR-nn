import torch.optim as optim
from lookahead import LookaheadRadam, LookaheadNovograd, LookaheadAdam
from novograd import Novograd
from radam import RAdam
import torch
import torch.nn as nn


class Optimizer(object):
    def __init__(self, parameters, config):
        self.config = config
        self.optimizer = build_optimizer(parameters, config)
        self.global_step = 1
        self.current_epoch = 0
        self.lr = config.lr
        self.decay_ratio = config.decay_ratio
        self.epoch_decay_flag = False

    def step(self):
        self.global_step += 1
        self.optimizer.step()

    def epoch(self):
        self.current_epoch += 1

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def decay_lr(self):
        self.lr *= self.decay_ratio
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr


def build_optimizer(parameters, config):
    if config.type == 'adam':
        return optim.Adam(
            parameters,
            lr=config.lr,
            betas=(0.9, 0.98),
            eps=1e-08,
            weight_decay=config.weight_decay
        )
    elif config.type == 'sgd':
        return optim.SGD(
            params=parameters,
            lr=config.lr,
            momentum=config.momentum,
            nesterov=config.nesterov,
            weight_decay=config.weight_decay
        )
    elif config.type == 'adadelta':
        return optim.Adadelta(
            params=parameters,
            lr=config.lr,
            rho=config.rho,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.type == 'LookaheadRadam':
        return LookaheadRadam(params=parameters,
                              lr=config.lr,
                              betas=(0.9, 0.999),
                              eps=1e-8,
                              weight_decay=config.weight_decay)
    elif config.type == 'Radam':
        return RAdam(params=parameters,
                     lr=config.lr,
                     betas=(0.9, 0.999),
                     eps=1e-8,
                     weight_decay=config.weight_decay)
    elif config.type == 'LookaheadAdam':
        return LookaheadAdam(params=parameters,
                             lr=config.lr,
                             betas=(0.9, 0.999),
                             eps=1e-8,
                             weight_decay=config.weight_decay)
    elif config.type == 'LookaheadNovograd':
        return LookaheadNovograd(params=parameters,
                                 lr=config.lr,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 weight_decay=config.weight_decay)
    elif config.type == 'Novograd':
        return Novograd(params=parameters,
                        lr=config.lr,
                        betas=(0.9, 0.999),
                        eps=1e-8,
                        weight_decay=config.weight_decay)
    # elif config.type == 'Ranger':
    #     return Ranger(params=parameters,
    #                   lr=config.lr,
    #                   betas=(0.9, 0.999),
    #                   eps=1e-8,
    #                   weight_decay=config.weight_decay)
    else:
        raise NotImplementedError


def label_smoothing(label, n_class, smoothing=0.1):
    '''
    :param label: torch.Size([N]) or torch.Size([N,1])
    :param n_class: number of classes
    :param smoothing: probability of converting current label to other labels
    :return: label-smoothed one-hot labels, shape of torch.Size([N,n_class])
    '''
    assert label.ndim == 1 or label.ndim == 2

    if label.ndim == 1:
        label = label.unsqueeze(1)

    batch_size = label.shape[0]

    y_one_hot = torch.zeros(batch_size, n_class).scatter_(1, label, 1)

    label_smooth = ((1 - smoothing) * y_one_hot) + (smoothing / (n_class - 1)) * (
            1 - y_one_hot)  # [batch_size, n_class]

    return label_smooth


# class LabelSmoothingLoss(nn.Module):
#     def __init__(self, classes, smoothing=0.0, dim=-1):
#         super(LabelSmoothingLoss, self).__init__()
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.cls = classes
#         self.dim = dim
#
#     def forward(self, pred, target):
#         pred = pred.log_softmax(dim=self.dim)
#         with torch.no_grad():
#             # true_dist = pred.data.clone()
#             true_dist = torch.zeros_like(pred)
#             true_dist.fill_(self.smoothing / (self.cls - 1))
#             true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#         return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, class_weights=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.class_weights = class_weights if class_weights is not None else torch.ones(classes)

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        # Apply class weights
        class_weights = self.class_weights[target.data].unsqueeze(1)
        loss = torch.sum(-true_dist * pred, dim=self.dim) * class_weights.squeeze()

        return torch.mean(loss)