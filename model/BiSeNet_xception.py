import torch
from torch import nn
from .xception import xception
import torch.nn.functional as F


class SpatialPath(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = self.downsample_block(3, 64)
        self.layer2 = self.downsample_block(64, 128)
        self.layer3 = self.downsample_block(128, 256)

    def downsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)


class ARM(nn.Module):

    def __init__(self, input_h, input_w, channels):
        super().__init__()
        self.pool = nn.AvgPool2d( (input_h, input_w) )
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        feature_map = x
        x = self.pool(x)
        x = self.conv(x)
        x = self.norm(x)
        x = torch.sigmoid(x)
        return x.expand_as(feature_map) * feature_map


class FFM(nn.Module):

    def __init__(self, input_h, input_w, channels):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.pool = nn.AvgPool2d( ( input_h // 8, input_w // 8 ))
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        feature = torch.cat([x1, x2], dim=1)
        feature = self.feature(feature)

        x = self.pool(feature)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return feature + x.expand_as(feature) * feature


class ContextPath(nn.Module):

    def __init__(self, input_h, input_w):
        super().__init__()
        self.input_h = input_h
        self.input_w = input_w
        self.backbone = xception()
        self.x8_arm = ARM(input_h // 8, input_w // 8, 256)
        self.x16_arm = ARM(input_h // 16, input_w // 16, 728)
        self.x32_arm = ARM(input_h // 32, input_w // 32,  2048)
        self.global_pool = nn.AvgPool2d((input_h // 32, input_w //32))

    def forward(self, x):
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)

        feature_x8 = self.backbone.layer3(x)
        feature_x16 = self.backbone.layer4(feature_x8)
        feature_x32 = self.backbone.layer5(feature_x16)
        center = self.global_pool(feature_x32)

        feature_x8 = self.x8_arm(feature_x8)
        feature_x16 = self.x16_arm(feature_x16)
        feature_x32 = self.x32_arm(feature_x32)

        up_feature_x32 = F.upsample(center, size=(self.input_h // 32, self.input_w //32), mode='bilinear', align_corners=False)
        ensemble_feature_x32 = feature_x32 + up_feature_x32

        up_feature_x16 = F.upsample(ensemble_feature_x32, scale_factor=2, mode='bilinear', align_corners=False)
        ensemble_feature_x16 = torch.cat((feature_x16, up_feature_x16), dim=1)

        up_feature_x8 = F.upsample(ensemble_feature_x16, scale_factor=2, mode='bilinear', align_corners=False)
        ensemble_feature_x8 = torch.cat((feature_x8, up_feature_x8), dim=1)

        return ensemble_feature_x8


class BiSeNet_Xception34(nn.Module):

    def __init__(self, input_h, input_w, n_classes = 19):
        super().__init__()
        self.spatial_path = SpatialPath()
        self.context_path = ContextPath(input_h, input_w)
        self.ffm = FFM(input_h, input_w, 3288)
        self.pred = nn.Conv2d(3288, n_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.spatial_path(x)
        #print("x1.size:", x1.size())
        x2 = self.context_path(x)
        #print("x2.size:", x2.size())
        feature = self.ffm(x1, x2)
        seg = self.pred(feature)
        return F.upsample(seg, x.size()[2:], mode='bilinear', align_corners=False)


if __name__ == '__main__':
    #images = torch.rand(2, 3, 224, 224)
    images = torch.rand(2, 3, 1024, 2048)
    model = BiSeNet_Xception34(1024, 2048)
    print(model(images).size())
