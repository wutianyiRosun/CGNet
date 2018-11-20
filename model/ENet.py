import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

__all__ = ["ENet"]


class InitialBlock(nn.Module):
    '''
    The initial block for Enet has 2 branches: The convolution branch and
    maxpool branch.
    The conv branch has 13 layers, while the maxpool branch gives 3 layers
    corresponding to the RBG channels.
    Both output layers are then concatenated to give an output of 16 layers.
    INPUTS:
    - input(Tensor): A 4D tensor of shape [batch_size, channel, height, width]
    '''

    def __init__(self):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(3, 13, (3, 3), stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(13, 1e-3)
        self.prelu = nn.PReLU(13)
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, input):
        output = torch.cat([
            self.prelu(self.batch_norm(self.conv(input))), self.pool(input)
        ], 1)
        return output


class BottleNeck(nn.Module):
    '''
    The bottle module has three different kinds of variants:
    1. A regular convolution which you can decide whether or not to downsample.
    2. A dilated convolution which requires you to have a dilation factor.
    3. An asymetric convolution that has a decomposed filter size of 5x1 and
    1x5 separately.
    INPUTS:
    - inputs(Tensor): a 4D Tensor of the previous convolutional block of shape
    [batch_size, channel, height, widht].
    - output_channels(int): an integer indicating the output depth of the
    output convolutional block.
    - regularlizer_prob(float): the float p that represents the prob of
    dropping a layer for spatial dropout regularlization.
    - downsampling(bool): if True, a max-pool2D layer is added to downsample
    the spatial sizes.
    - upsampling(bool): if True, the upsampling bottleneck is activated but
    requires pooling indices to upsample.
    - dilated(bool): if True, then dilated convolution is done, but requires
    a dilation rate to be given.
    - dilation_rate(int): the dilation factor for performing atrous
    convolution/dilated convolution
    - asymmetric(bool): if True, then asymmetric convolution is done, and
    the only filter size used here is 5.
    - use_relu(bool): if True, then all the prelus become relus according to
    Enet author.
    '''

    def __init__(self,
                 input_channels=None,
                 output_channels=None,
                 regularlizer_prob=0.1,
                 downsampling=False,
                 upsampling=False,
                 dilated=False,
                 dilation_rate=None,
                 asymmetric=False,
                 use_relu=False):
        super(BottleNeck, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.downsampling = downsampling
        self.upsampling = upsampling
        self.use_relu = use_relu

        internal = output_channels // 4
        input_stride = 2 if downsampling else 1
        # First projection with 1x1 kernel (2x2 for downsampling)
        conv1x1_1 = nn.Conv2d(input_channels, internal,
                              input_stride, input_stride, bias=False)
        batch_norm1 = nn.BatchNorm2d(internal, 1e-3)
        prelu1 = self._prelu(internal, use_relu)
        self.block1x1_1 = nn.Sequential(conv1x1_1, batch_norm1, prelu1)

        conv = None
        if downsampling:
            self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
            conv = nn.Conv2d(internal, internal, 3, stride=1, padding=1)
        elif upsampling:
            # padding is replaced with spatial convolution without bias.
            spatial_conv = nn.Conv2d(input_channels, output_channels, 1,
                                     bias=False)
            batch_norm = nn.BatchNorm2d(output_channels, 1e-3)
            self.conv_before_unpool = nn.Sequential(spatial_conv, batch_norm)
            self.unpool = nn.MaxUnpool2d(2)
            conv = nn.ConvTranspose2d(internal, internal, 3,
                                      stride=2, padding=1, output_padding=1)
        elif dilated:
            conv = nn.Conv2d(internal, internal, 3, padding=dilation_rate,
                             dilation=dilation_rate)
        elif asymmetric:
            conv1 = nn.Conv2d(internal, internal, [5, 1], padding=(2, 0),
                              bias=False)
            conv2 = nn.Conv2d(internal, internal, [1, 5], padding=(0, 2))
            conv = nn.Sequential(conv1, conv2)
        else:
            conv = nn.Conv2d(internal, internal, 3, padding=1)

        batch_norm = nn.BatchNorm2d(internal, 1e-3)
        prelu = self._prelu(internal, use_relu)
        self.middle_block = nn.Sequential(conv, batch_norm, prelu)

        # Final projection with 1x1 kernel
        conv1x1_2 = nn.Conv2d(internal, output_channels, 1, bias=False)
        batch_norm2 = nn.BatchNorm2d(output_channels, 1e-3)
        prelu2 = self._prelu(output_channels, use_relu)
        self.block1x1_2 = nn.Sequential(conv1x1_2, batch_norm2, prelu2)

        # regularlize
        self.dropout = nn.Dropout2d(regularlizer_prob)

    def _prelu(self, channels, use_relu):
        return (nn.PReLU(channels) if use_relu is False else nn.ReLU())

    def forward(self, input, pooling_indices=None):
        main = None
        input_shape = input.size()
        if self.downsampling:
            main, indices = self.pool(input)
            if (self.output_channels != self.input_channels):
                pad = Variable(torch.Tensor(input_shape[0],
                               self.output_channels - self.input_channels,
                               input_shape[2] // 2,
                               input_shape[3] // 2).zero_(), requires_grad=False)
                if (torch.cuda.is_available):
                    pad = pad.cuda(0)
                main = torch.cat((main, pad), 1)
        elif self.upsampling:
            main = self.unpool(self.conv_before_unpool(input), pooling_indices)
        else:
            main = input

        other_net = nn.Sequential(self.block1x1_1, self.middle_block,
                                  self.block1x1_2)
        other = other_net(input)
        output = F.relu(main + other)
        if (self.downsampling):
            return output, indices
        return output

ENCODER_LAYER_NAMES = ['initial', 'bottleneck_1_0', 'bottleneck_1_1',
                       'bottleneck_1_2', 'bottleneck_1_3', 'bottleneck_1_4',
                       'bottleneck_2_0', 'bottleneck_2_1', 'bottleneck_2_2',
                       'bottleneck_2_3', 'bottleneck_2_4', 'bottleneck_2_5',
                       'bottleneck_2_6', 'bottleneck_2_7', 'bottleneck_2_8',
                       'bottleneck_3_1', 'bottleneck_3_2', 'bottleneck_3_3',
                       'bottleneck_3_4', 'bottleneck_3_5', 'bottleneck_3_6',
                       'bottleneck_3_7', 'bottleneck_3_8', 'classifier']
DECODER_LAYER_NAMES = ['bottleneck_4_0', 'bottleneck_4_1', 'bottleneck_4_2'
                       'bottleneck_5_0', 'bottleneck_5_1', 'fullconv']


class Encoder(nn.Module):
    def __init__(self, num_classes, only_encode=True):
        super(Encoder, self).__init__()
        self.state = only_encode
        layers = []
        layers.append(InitialBlock())
        layers.append(BottleNeck(16, 64, regularlizer_prob=0.01,
                                 downsampling=True))
        for i in range(4):
            layers.append(BottleNeck(64, 64, regularlizer_prob=0.01))
        
        # Section 2 and 3
        layers.append(BottleNeck(64, 128, downsampling=True))
        for i in range(2):
            layers.append(BottleNeck(128, 128))
            layers.append(BottleNeck(128, 128, dilated=True, dilation_rate=2))
            layers.append(BottleNeck(128, 128, asymmetric=True))
            layers.append(BottleNeck(128, 128, dilated=True, dilation_rate=4))
            layers.append(BottleNeck(128, 128))
            layers.append(BottleNeck(128, 128, dilated=True, dilation_rate=8))
            layers.append(BottleNeck(128, 128, asymmetric=True))
            layers.append(BottleNeck(128, 128, dilated=True, dilation_rate=16))
        # only training encoder
        if only_encode:
            layers.append(nn.Conv2d(128, num_classes, 1))

        for layer, layer_name in zip(layers, ENCODER_LAYER_NAMES):
            super(Encoder, self).__setattr__(layer_name, layer)
        self.layers = layers

    
    def forward(self, input):
        pooling_stack = []
        output = input
        for layer in self.layers:
            if hasattr(layer, 'downsampling') and layer.downsampling:
                output, pooling_indices = layer(output)
                pooling_stack.append(pooling_indices)
            else:
                output = layer(output)

        if self.state:
            output = F.upsample(output, cfg.TRAIN.IMG_SIZE, None, 'bilinear')

        return output, pooling_stack


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        layers = []
        # Section 4
        layers.append(BottleNeck(128, 64, upsampling=True, use_relu=True))
        layers.append(BottleNeck(64, 64, use_relu=True))
        layers.append(BottleNeck(64, 64, use_relu=True))

        # Section 5
        layers.append(BottleNeck(64, 16, upsampling=True, use_relu=True))
        layers.append(BottleNeck(16, 16, use_relu=True))
        layers.append(nn.ConvTranspose2d(16, num_classes, 2, stride=2))

        self.layers = nn.ModuleList([layer for layer in layers])
    
    def forward(self, input, pooling_stack):
        output = input
        for layer in self.layers:
            if hasattr(layer, 'upsampling') and layer.upsampling:
                pooling_indices = pooling_stack.pop()
                output = layer(output, pooling_indices)
            else:
                output = layer(output)
        return output


class ENet(nn.Module):
    def __init__(self,n_classes=19, only_encode=False):
        super(ENet, self).__init__()
        self.state = only_encode
        self.encoder = Encoder(n_classes,only_encode=only_encode)
        self.decoder = Decoder(n_classes)

    def forward(self, input):
        output, pooling_stack = self.encoder(input)
        if not self.state:
            output = self.decoder(output, pooling_stack)
        return output
