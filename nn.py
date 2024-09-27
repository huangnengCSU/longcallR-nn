import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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

class SPPLayer(nn.Module):
    def __init__(self, pool_sizes):
        super(SPPLayer, self).__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        batch_size, c, h, w = x.size()
        spp_output = []
        for pool_size in self.pool_sizes:
            pooled = F.adaptive_max_pool2d(x, output_size=(pool_size, pool_size))
            spp_output.append(pooled.reshape(batch_size, -1))
        spp_output = torch.cat(spp_output, dim=1)
        return spp_output


class ResNetwork(nn.Module):
    def __init__(self, config):
        super(ResNetwork, self).__init__()
        self.config = config

        # Select the correct ResNet model
        if config.model_name == 'resnet18':
            self.resnet = models.resnet18(pretrained=config.pretrained)
        elif config.model_name == 'resnet34':
            self.resnet = models.resnet34(pretrained=config.pretrained)
        elif config.model_name == 'resnet50':
            self.resnet = models.resnet50(pretrained=config.pretrained)
        elif config.model_name == 'resnet101':
            self.resnet = models.resnet101(pretrained=config.pretrained)
        elif config.model_name == 'resnet152':
            self.resnet = models.resnet152(pretrained=config.pretrained)
        else:
            raise ValueError("Unexpected model name")

        # Adjust the first convolutional layer to accept 7 input channels
        self.resnet.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if config.spp:
            pool_sizes = [1, 2, 4]
            self.resnet.avgpool = nn.Identity()
            self.resnet.fc = nn.Identity()
            self.spp = SPPLayer(pool_sizes)
            # Compute number of features after SPP
            num_features = self.resnet.layer4[2].conv3.out_channels * sum([size * size for size in pool_sizes])
            self.zy_fc = nn.Linear(num_features, config.num_zy_class, bias=True)
            self.gt_fc = nn.Linear(num_features, config.num_gt_class, bias=True)
        else:
            self.zy_fc = nn.Linear(self.resnet.fc.in_features, config.num_zy_class, bias=True)
            self.gt_fc = nn.Linear(self.resnet.fc.in_features, config.num_gt_class, bias=True)

        self.zy_crit = LabelSmoothingLoss(config.num_zy_class, smoothing=config.smoothing)
        if config.use_gt_class_weight:
            gt_class_weights = torch.tensor(
                [1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 3.0, 1.0])
            self.gt_crit = LabelSmoothingLoss(config.num_gt_class, 0.1, class_weights=gt_class_weights)
        else:
            self.gt_crit = LabelSmoothingLoss(config.num_gt_class, smoothing=config.smoothing)

    def forward(self, x, zy_target, gt_target):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        if self.config.spp:
            x = self.spp(x)
        else:
            x = self.resnet.avgpool(x)
            x = x.reshape(x.size(0), -1)  # Use reshape instead of flatten

        zy_logits = self.zy_fc(x)
        zy_loss = self.zy_crit(zy_logits.contiguous().view(-1, self.config.num_zy_class),
                               zy_target.contiguous().view(-1))

        gt_logits = self.gt_fc(x)
        gt_loss = self.gt_crit(gt_logits.contiguous().view(-1, self.config.num_gt_class),
                               gt_target.contiguous().view(-1))
        loss = 2.0 * gt_loss + zy_loss
        return loss, zy_logits, gt_logits

    def predict(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        if self.config.spp:
            x = self.spp(x)
        else:
            x = self.resnet.avgpool(x)
            x = x.reshape(x.size(0), -1)  # Use reshape instead of flatten

        zy_logits = self.zy_fc(x)
        zy_probs = torch.softmax(zy_logits, dim=1)

        gt_logits = self.gt_fc(x)
        gt_probs = torch.softmax(gt_logits, dim=1)
        return zy_probs, gt_probs
