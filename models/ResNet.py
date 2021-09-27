from torch import nn


class basic_block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(basic_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=downsample + 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        if downsample:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=downsample + 1, padding=1)
        self.downsample = downsample

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn(x1)
        if self.downsample:
            x2 = self.conv3(x)
        else:
            x2 = x
        y = x1 + x2
        y = self.relu(y)
        return y


class ResNet_plane(nn.Module):
    def __init__(self, in_channels=3):
        super(ResNet_plane, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2,
                               padding=3)  # 好的吧，不准在有stride的情況下使用padding='same'
        # 在复现过程中回看知乎才发现，第一个conv之后也立马跟了一个BN
        self.bn = nn.BatchNorm2d(64)
        self.maxpooling = nn.MaxPool2d(kernel_size=7, stride=2, padding=3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.maxpooling(x)
        return x


class ResNet_on_basic(nn.Module):
    def __init__(self, block_nums: list = [2, 2, 2, 2], in_channels=3, gap_end=False):
        super(ResNet_on_basic, self).__init__()
        self.plane = ResNet_plane(in_channels=in_channels)
        self.in_channels = 64
        self.out_channels = 64
        layers = []
        for i, num in enumerate(block_nums):
            layer = self.layer_generate(num, i)
            layers.append(layer)
        self.body = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.out_channels, 1000)

        self.gap_end = gap_end

    def layer_generate(self, block_num, layer_index):
        layer = []
        for i in range(block_num):
            if i == 0:  # 如果是每个layer的第0个conv
                if layer_index >= 1:
                    self.out_channels = 2 * self.in_channels
                    downsample = True
                else:  # 如果是第0个layer，则不做下采样
                    downsample = False
            else:  # 如果是每层的其他conv
                self.in_channels = self.out_channels
                downsample = False
            block = basic_block(self.in_channels, self.out_channels, downsample=downsample)
            layer.append(block)
            self.in_channels = self.out_channels
        layer = nn.Sequential(*layer)
        return layer

    def forward(self, x):
        x = self.plane(x)
        x = self.body(x)
        x = self.gap(x)
        if not self.gap_end:
            x = x.reshape(x.shape[0], -1)
            x = self.fc(x)
        return x


def ResNet18(in_channles=3, gap_end=False):
    return ResNet_on_basic(in_channels=in_channles, gap_end=gap_end)


def ResNet34(in_channels=3, gap_end=False):
    return ResNet_on_basic([3, 4, 6, 3], in_channels=in_channels, gap_end=gap_end)


if __name__ == '__main__':
    import torch
    # import numpy as np
    from torchsummary import torchsummary

    model = ResNet34()
    if torch.cuda.device_count():
        summary_device = 'cuda'
    else:
        summary_device = 'cuda'
    model.to(summary_device)
    torchsummary.summary(model, (3, 224, 224), device=summary_device)
    # data = np.zeros((1,3, 224,224))
    # tensor = torch.Tensor(data)
    # output = model(tensor)
    # import torchvision
    # torchvision.models.ResNet
