import torch.nn as nn
import torch


class net(nn.Module):

    def __init__(self, in_dim, dilate_rate, kernel_1=(5, 2), kernel_2=(1, 1), kernel_3=(3, 2)):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, 32, kernel_size=kernel_1, dilation=(1, dilate_rate), bias=False)
        self.conv1.weight.data = nn.init.trunc_normal_(self.conv1.weight.data, std=0.1, a=-0.2, b=0.2)
        self.conv2 = nn.Conv2d(32, 8, kernel_size=kernel_2, dilation=(1, dilate_rate), bias=False)
        self.conv2.weight.data = nn.init.trunc_normal_(self.conv2.weight.data, std=0.1, a=-0.2, b=0.2)
        self.conv3 = nn.Conv2d(8, dilate_rate - 1, kernel_size=kernel_3, dilation=(1, dilate_rate), bias=False)
        self.conv3.weight.data = nn.init.trunc_normal_(self.conv3.weight.data, std=0.1, a=-0.2, b=0.2)
        self.act = nn.ReLU()

        self.grp_conv = nn.Conv2d(in_dim, dilate_rate - 1, kernel_size=(5, 2), dilation=(1, dilate_rate), bias=False)
        self.grp_conv.weight.data = nn.init.trunc_normal_(self.grp_conv.weight.data, std=0.1, a=-0.2, b=0.2)
        self.dilate_rate = dilate_rate

    def forward(self, x):

        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.conv3(h)

        grapRes = self.grp_conv(x)
        grapRes = grapRes[:, :, 1:-1, :-self.dilate_rate]
        return h, grapRes


class rRAKI(nn.Module):

    def __init__(self, in_dim, dilate_rate, kernel_1=(5, 2), kernel_2=(1, 1), kernel_3=(3, 2)):
        super().__init__()

        network = []
        for i in range(in_dim):
            network.append(net(in_dim, dilate_rate, kernel_1, kernel_2, kernel_3))
        self.network = nn.ModuleList(network)

    def forward(self, x):
        out, res = [], []
        for i, layer in enumerate(self.network):
            h, grapRes = layer(x)
            out.append(h.unsqueeze(0))
            res.append(grapRes.unsqueeze(0))
        return torch.cat(out, dim=0), torch.cat(res, dim=0)


class rRAKI_loss(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, y):
        return torch.norm(y - x[0] - x[1], p=2) + torch.norm(y - x[1], p=2)

