import torch
import torch.nn as nn


# 基础卷积定义
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


# 像素注意力模块（Pixel Attention Layer）
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


# 通道注意力模块（Channel Attention Layer）
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


# 多尺度特征融合模块（Multi-Scale Feature Fusion Module）
class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=7, padding=3)
        self.fuse = nn.Conv2d((in_channels // 2) * 3 + in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x_concat = torch.cat([x1, x2, x3, x], dim=1)
        return self.fuse(x_concat)


# 自适应残差结构（Adaptive Residual Block）
class AdaptiveResidualBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(AdaptiveResidualBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res = self.scale * res
        return res + x


# Group 结构定义
class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [AdaptiveResidualBlock(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)
        self.msfm = MultiScaleFeatureFusion(dim, dim)

    def forward(self, x):
        res = self.gp(x)
        res = self.msfm(res)
        return res + x


class HERA_Modified(nn.Module):
    def __init__(self, gps=3, blocks=1, conv=default_conv):
        super(HERA_Modified, self).__init__()
        self.gps = gps
        self.dim = 16  # 增加通道数以捕捉更多信息
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]

        # Group 结构
        self.groups = nn.ModuleList([Group(conv, self.dim, kernel_size, blocks) for _ in range(gps)])

        # 通道注意力和像素注意力
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * gps, self.dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 4, self.dim * gps, 1),
            nn.Sigmoid()
        )
        self.palayer = PALayer(self.dim)

        post_process = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)
        ]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_process)

    def forward(self, x1):
        x = self.pre(x1)

        # 使用 Group 结构并融合特征
        group_outs = [group(x) for group in self.groups]
        w = self.ca(torch.cat(group_outs, dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = sum(w[:, i] * group_outs[i] for i in range(self.gps))

        # 像素注意力处理
        out = self.palayer(out)
        x = self.post(out)

        return x + x1


# 测试代码
if __name__ == "__main__":
    image_size = (1, 3, 640, 640)
    image = torch.rand(*image_size)
    net = HERA_Modified(gps=3, blocks=1)
    out = net(image)
    print(out.size())