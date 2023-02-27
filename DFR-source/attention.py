
import numpy as np
import torch
from torch import nn
from torch.nn import init
import math
class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局均值池化  输出的是c×1×1
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # channel // reduction代表通道压缩
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  # 还原
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            print(m)  # 没运行到这儿
            if isinstance(m, nn.Conv2d):  # 判断类型函数——：m是nn.Conv2d类吗？
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()  # 50×512×7×7
        y = self.avg_pool(x).view(b, c)  # ① maxpool之后得：50×512×1×1 ② view形状得到50×512
        y = self.fc(y).view(b, c, 1, 1)  # 50×512×1×1
        return x * y.expand_as(x)  # 根据x.size来扩展y


class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
