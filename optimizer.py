import math
from torch.optim import Optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= warmup:
            raise ValueError("Invalid warmup steps: {}".format(warmup))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, warmup=warmup)
        super(RAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                t = state['step']

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected first and second moment estimates
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * t * beta2 ** t / bias_correction2

                if N_sma >= 5:
                    rect = math.sqrt(((N_sma - 4) * (N_sma - 2) * N_sma_max) /
                                     ((N_sma_max - 4) * (N_sma_max - 2) * N_sma))
                    lr = group['lr'] * rect / bias_correction1
                else:
                    lr = group['lr'] / bias_correction1

                # Warmup
                if group['warmup'] > 0 and t <= group['warmup']:
                    lr *= t / group['warmup']

                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * lr)

                # Calculate step size
                denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2) + group['eps']

                # Debugging statements
                print(f"Denominator tensor: {denom}")
                print(f"Denominator tensor shape: {denom.shape}")

                # Ensure step_size is a scalar by computing per element scale
                step_size = lr / denom

                # Debugging statements
                print(f"Step size tensor: {step_size}")
                print(f"Step size tensor shape: {step_size.shape}")

                # Ensure step_size is a scalar
                if step_size.numel() == 1:
                    step_size = step_size.item()  # Convert tensor to scalar if it's a single-element tensor
                else:
                    raise ValueError("step_size is not a scalar. Ensure proper calculation.")

                # Apply the step
                p.data.add_(exp_avg, alpha=-step_size)

        return loss


class Lookahead(Optimizer):
    def __init__(self, optimizer, alpha=0.5, k=5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Invalid alpha parameter: {}".format(alpha))
        if not 1 <= k:
            raise ValueError("Invalid k parameter: {}".format(k))
        self.optimizer = optimizer
        self.alpha = alpha
        self.k = k
        self.state = {}

        for group in optimizer.param_groups:
            for p in group['params']:
                self.state[p] = dict(
                    step=0,
                    slow_buffer=torch.empty_like(p.data)
                )
                self.state[p]['slow_buffer'].copy_(p.data)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # Perform a single optimization step
        loss = self.optimizer.step(closure)

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['step'] += 1

                # Copy the slow parameters if it is the k-th step
                if state['step'] % self.k == 0:
                    slow = state['slow_buffer']
                    slow.add_(p.data - slow, alpha=self.alpha)
                    p.data.copy_(slow)

        return loss

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)



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
        device = pred.device
        self.class_weights = self.class_weights.to(device)
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        # Apply class weights
        class_weights = self.class_weights[target.data].unsqueeze(1)
        loss = torch.sum(-true_dist * pred, dim=self.dim) * class_weights.squeeze()

        return torch.mean(loss)


class FocalLoss(nn.Module):
    def __init__(self, classes, alpha=1.0, gamma=2.0, reduction='mean', class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.classes = classes
        self.class_weights = class_weights if class_weights is not None else torch.ones(classes)

    def forward(self, inputs, targets):
        device = inputs.device
        self.class_weights = self.class_weights.to(device)

        # Compute softmax probabilities
        probs = F.softmax(inputs, dim=-1)

        # Gather the probabilities of the target classes
        target_probs = probs.gather(1, targets.view(-1, 1)).squeeze()

        # Compute the cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute focal loss component
        focal_loss_component = (1 - target_probs) ** self.gamma

        # Compute weighted focal loss
        focal_loss = self.alpha * focal_loss_component * ce_loss

        # Apply class weights
        class_weights = self.class_weights[targets].to(device)
        focal_loss *= class_weights

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

