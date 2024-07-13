import torch
import torch.nn as nn
import torchvision.models as models
from optim import LabelSmoothingLoss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class BaseEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout=0.2, bidirectional=True):
        super(BaseEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.output_proj = nn.Linear(2 * hidden_size if bidirectional else hidden_size,
                                     output_size,
                                     bias=True)

    def forward(self, inputs):
        assert inputs.dim() == 3

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(inputs)

        logits = self.output_proj(outputs)  # N, L, output_size

        return logits, hidden


def build_encoder(config):
    if config.enc.type == 'lstm':
        return BaseEncoder(
            input_size=config.feature_dim,
            hidden_size=config.enc.hidden_size,
            output_size=config.enc.output_size,
            n_layers=config.enc.n_layers,
            dropout=config.dropout,
            bidirectional=config.enc.bidirectional
        )
    else:
        raise NotImplementedError


class ForwardLayer(nn.Module):
    def __init__(self, input_size, inner_size, zy_class):
        super(ForwardLayer, self).__init__()
        self.dense = nn.Linear(input_size, inner_size, bias=True)
        self.tanh = nn.Tanh()
        self.zygosity_layer = nn.Linear(inner_size, zy_class, bias=True)

    def forward(self, inputs):
        out = self.tanh(self.dense(inputs))  # [batch, length, hidden*2]
        out = out[:, 16, :]  # [batch, hidden*2]
        zy_outputs = self.zygosity_layer(out)
        return zy_outputs


def build_forward(config):
    return ForwardLayer(config.enc.output_size,
                        config.joint.inner_size,
                        config.num_class)


class LSTMNetwork(nn.Module):
    def __init__(self, config):
        super(LSTMNetwork, self).__init__()
        # define encoder
        self.config = config
        self.encoder = build_encoder(config)
        self.forward_layer = build_forward(config)
        self.zy_crit = LabelSmoothingLoss(config.num_class, 0.1)

    def forward(self, inputs, zy_target):
        # inputs: N x L x c
        enc_state, _ = self.encoder(inputs)  # [N, L, o]
        zy_logits = self.forward_layer(enc_state)
        zy_loss = self.zy_crit(zy_logits.contiguous().view(-1, self.config.num_class),
                               zy_target.contiguous().view(-1))
        loss = zy_loss

        return loss, zy_logits

    def predict(self, inputs):
        enc_state, _ = self.encoder(inputs)  # [N, L, o]
        zy_logits = self.forward_layer(enc_state)
        zy_logits = torch.softmax(zy_logits, 1)
        return zy_logits


class ResNetwork(nn.Module):
    def __init__(self, config):
        super(ResNetwork, self).__init__()
        self.config = config
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)  # change input channel
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, config.num_class, bias=True)  # change output class
        self.zy_crit = LabelSmoothingLoss(config.num_class, 0.1)

    def forward(self, inputs, zy_target):
        zy_logits = self.resnet18(inputs)
        zy_loss = self.zy_crit(zy_logits.contiguous().view(-1, self.config.num_class),
                               zy_target.contiguous().view(-1))
        loss = zy_loss
        return loss, zy_logits

    def predict(self, inputs):
        zy_logits = self.resnet18(inputs)
        zy_logits = torch.softmax(zy_logits, 1)
        return zy_logits
