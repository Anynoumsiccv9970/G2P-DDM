"""
Implementation of "Fully Convolutional Networks for Continuous Sign Language Recognition"
"""

import torch
import torch.nn as nn
# from .local_attn import Encoder, mask_local_mask, LayerNorm
from .temporal_local_attention import TransformerEncoder

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride=1)
        self.bn1 = self.norm_layer(planes, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # downsample
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = self.norm_layer(planes, affine=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class MainStream(nn.Module):
    def __init__(self, vocab_size, opts, momentum=0.1):
        super(MainStream, self).__init__()

        # cnn
        # first layer: channel 3 -> 32
        self.conv = conv3x3(3, 32)
        self.bn = nn.BatchNorm2d(32, momentum=momentum,  affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        # 4 basic blocks
        channels = [32, 64, 128, 256, 512]
        layers = []
        for num_layer in range(len(channels) - 1):
            layers.append(BasicBlock(channels[num_layer], channels[num_layer + 1]))
        self.layers = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        self.conv1 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.enc1 = nn.Sequential(self.conv1, self.bn1, self.elu, self.pool1)

        # self-attention
        self.selfattn = TransformerEncoder(opts)

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.enc2 = nn.Sequential(self.conv2, self.bn2, self.elu, self.pool2)


        self.fc = nn.Linear(512, vocab_size)

        self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                m.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()


    def forward(self, video, len_video=None):
        """
        x: [batch, num_f, 3, h, w]
        """
        # print("input: ", video.size())
        bs, num_f, c, h, w = video.size()

        x = video.reshape(-1, c, h, w)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.layers(x)
        x = self.avgpool(x).squeeze_()  # [bs*t, 512]

        x = x.reshape(bs, -1, 512)     # [bs, t ,512]
        x = x.permute(0, 2, 1)  # [bs, 512, t]
        x = self.enc1(x)       # [bs, 512, t/2]
        x = self.enc2(x)       # [bs, 1024, t/4]
        x = x.permute(0, 2, 1)  # [bs, t/2, 512]
        
        padding_mask = self._get_mask(len_video/4).to(x.device)
        x = self.selfattn(x, mask=padding_mask, src_length=len_video/4)        # [bs, t/2, 512]

        x = x.permute(0, 2, 1)  # [bs, 512, t/2]
        x = x.permute(0, 2, 1)  # [bs, t/2, 512]
        logits = self.fc(x)  # [batch, t/4, vocab_size]
        return logits

    def _get_mask(self, x_len):
        pos = torch.arange(0, max(x_len)).unsqueeze(0).repeat(x_len.size(0), 1).to(x_len.device)
        pos[pos >= x_len.unsqueeze(1)] = max(x_len) + 1
        mask = pos.ne(max(x_len) + 1)
        return mask.unsqueeze(1)



if __name__ == "__main__":
    from config.options import parse_args
    opts = parse_args()
    x = torch.randn(2, 30, 3, 112, 112).cuda()
    len_x = torch.LongTensor([30, 20])
    model = MainStream(1233, opts).cuda()
    out = model(x, len_x)
    print("out: ", out.shape)
