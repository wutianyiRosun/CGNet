""" 
Creates an Xception Model as defined in:
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
This weights ported from the Keras implementation. Achieves the following performance on the validation set:
Loss:0.9173 Prec@1:78.892 Prec@5:94.292
REMEMBER to set your image size to 3x299x299 for both test and validation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch

__all__ = ['xception']

model_urls = {
    'xception':'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None
        
        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x



class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=19):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        
        self.num_classes = num_classes

        self.layer1 = nn.Sequential( nn.Conv2d(3, 32, 3,2, 0, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(32,64,3,bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))  # H/2 x W/2
        #do relu here

        self.layer2 = Block(64,128,2,2,start_with_relu=False,grow_first=True) # H/4 x W/4     # block1
        self.layer3 = Block(128,256,2,2,start_with_relu=True,grow_first=True) # H/8 x W/8     # block2

        self.layer4 = nn.Sequential( Block(256,728,2,2,start_with_relu=True,grow_first=True), # block3
                                     Block(728,728,3,1,start_with_relu=True,grow_first=True),
                                     Block(728,728,3,1,start_with_relu=True,grow_first=True),
                                     Block(728,728,3,1,start_with_relu=True,grow_first=True),
                                     Block(728,728,3,1,start_with_relu=True,grow_first=True),
                                     Block(728,728,3,1,start_with_relu=True,grow_first=True),
                                     Block(728,728,3,1,start_with_relu=True,grow_first=True),
                                     Block(728,728,3,1,start_with_relu=True,grow_first=True),  # block10
                                     Block(728,728,3,1,start_with_relu=True,grow_first=True))  # block11  H/16 x W/16

        self.layer5 = nn.Sequential( Block(728,1024,2,2,start_with_relu=True,grow_first=False),
                                     SeparableConv2d(1024,1536,3,1,1),
                                     nn.BatchNorm2d(1536),
                                     nn.ReLU(inplace=True),
                                     SeparableConv2d(1536,2048,3,1,1),
                                     nn.BatchNorm2d(2048),
                                     nn.ReLU(inplace=True)) #block12 H/32 x W/32

        self.fc = nn.Linear(2048, num_classes)



        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------





    def forward(self, x):
        x = self.layer1(x)  # H/2 x W/2
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.block4(x)  # H/16 x W/16
        x = self.block5(x)  # H/32 x W/32
        

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def xception(pretrained=False,**kwargs):
    """
    Construct Xception.
    """

    model = Xception(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['xception']))
    return model


def netParams(model):
    '''
    Computing total network parameters
    Args:
       model: model
    return: total network parameters
    '''
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters


if __name__ == '__main__':
    model = xception()
    print("model params:", netParams(model))

