"""
FaceNet model for embedding calculations
"""

import os

import torch
from torch import nn
from torch.nn import functional as F


class BasicConv2d(nn.Module):
    """
    BasicConv2d layer for InceptionResnetV1
    """

    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int,
                 stride: int, 
                 padding=0):
        super().__init__()

        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        ) # verify bias false

        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001, # value found in tensorflow
            momentum=0.1, # default pytorch value
            affine=True
        )

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Block35(nn.Module):
    """
    Block35 layer for InceptionResnetV1
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_0 = self.branch0(x)
        x_1 = self.branch1(x)
        x_2 = self.branch2(x)

        out = torch.cat((x_0, x_1, x_2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)

        return out


class Block17(nn.Module):
    """
    Block17 layer for InceptionResnetV1
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(128, 128, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )

        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_0 = self.branch0(x)
        x_1 = self.branch1(x)

        out = torch.cat((x_0, x_1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)

        return out


class Block8(nn.Module):
    """
    Block8 layer for InceptionResnetV1
    """

    def __init__(self, scale: float = 1.0, noReLU: bool = False):
        super().__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(192, 192, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )

        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)

        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_0 = self.branch0(x)
        x_1 = self.branch1(x)

        out = torch.cat((x_0, x_1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x

        if not self.noReLU:
            out = self.relu(out)

        return out


class Mixed_6a(nn.Module):
    """
    Mixed_6a layer for InceptionResnetV1
    """

    def __init__(self):
        super().__init__()

        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_0 = self.branch0(x)
        x_1 = self.branch1(x)
        x_2 = self.branch2(x)

        out = torch.cat((x_0, x_1, x_2), 1)

        return out


class Mixed_7a(nn.Module):
    """
    Mixed_6a layer for InceptionResnetV1
    """

    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_0 = self.branch0(x)
        x_1 = self.branch1(x)
        x_2 = self.branch2(x)
        x_3 = self.branch3(x)

        out = torch.cat((x_0, x_1, x_2, x_3), 1)

        return out


class InceptionResnetV1(nn.Module):
    """Inception Resnet V1 model.

    Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
    datasets.

    Keyword Arguments:
        pretrained {str} -- Optional pretraining dataset. Either "vggface2" or "casia-webface".
            (default: {None})
        classify {bool} -- Whether the model should output classification probabilities or feature
            embeddings. (default: {False})
        num_classes {int} -- Number of output classes. If "pretrained" is set and num_classes not
            equal to that used for the pretrained model, the final linear layer will be randomly
            initialized. (default: {None})
        dropout_prob {float} -- Dropout probability. (default: {0.6})
    """
    def __init__(self,
                 pretrained: str = "vggface2",
                 classify: bool = False,
                 num_classes: int = None,
                 dropout_prob: float = 0.6):

        super().__init__()

        # Set simple attributes
        self.pretrained = pretrained
        self.classify = classify
        self.num_classes = num_classes

        if pretrained == "vggface2":
            tmp_classes = 8631

        elif pretrained == "casia-webface":
            tmp_classes = 10575

        elif pretrained is None and self.num_classes is None:
            raise Exception("At least one of 'pretrained' or 'num_classes' must be specified")

        else:
            tmp_classes = self.num_classes

        # Define layers
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)

        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)

        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )

        self.mixed_6a = Mixed_6a()

        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )

        self.mixed_7a = Mixed_7a()

        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )

        self.block8 = Block8(noReLU=True)

        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)

        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

        self.logits = nn.Linear(512, tmp_classes)

        if pretrained is not None:
            features_path = os.path.join(os.path.dirname(__file__),
                                         f"weights/{pretrained}/features.pt")

            logits_path = os.path.join(os.path.dirname(__file__),
                                       f"weights/{pretrained}/logits.pt")

            state_dict = {}
            state_dict.update(torch.load(features_path))
            state_dict.update(torch.load(logits_path))

            self.load_state_dict(state_dict)

        if self.num_classes is not None:
            self.logits = nn.Linear(512, self.num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate embeddings or logits given a batch of input image tensors.

        Arguments:
            x {torch.tensor} -- Batch of image tensors representing faces.

        Returns:
            torch.tensor -- Batch of embedding vectors or multinomial logits.
        """
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)

        if self.classify:
            x = self.logits(x)

        else:
            x = F.normalize(x, p=2, dim=1)

        return x
