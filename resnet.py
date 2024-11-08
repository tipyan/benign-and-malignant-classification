import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
class conv_block(nn.Module):
    def __init__(self, infilter, outfilter):
        super().__init__()
        self.conv0 = nn.Conv2d(infilter, outfilter, 3, stride=1, padding='same')
        self.conv1 = nn.Conv2d(outfilter, outfilter, 3, stride=1, padding='same')
        self.batch_conv0 = nn.BatchNorm2d(outfilter)
        self.batch_conv1 = nn.BatchNorm2d(outfilter)

    def forward(self, inputs):
        x0 = self.conv0(inputs)
        x0 = self.batch_conv0(x0)
        x0 = torch.nn.functional.gelu(x0)
        x0 = self.conv1(x0)
        x0 = self.batch_conv1(x0)
        x0 = torch.nn.functional.gelu(x0)
        return x0
        
class ResNet50(nn.Module):
    def __init__(self, infilter, pretrained=True):
        super().__init__()
        self.resnet50 =timm.create_model('resnet50', pretrained=pretrained, features_only=True)
        self.block0 = conv_block(256+infilter*2, infilter)
        self.block1 = conv_block(512+infilter*4, infilter*2)
        self.upsample0 = nn.ConvTranspose2d(infilter*2, infilter*2, 3, stride=2, padding=1, output_padding=1)
        self.upsample1 = nn.ConvTranspose2d(infilter*4, infilter*4, 3, stride=2, padding=1, output_padding=1)
        #self.upsample0 = nn.Upsample(scale_factor=2, mode='nearest')
        #self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, inputs):
        reslist = self.resnet50(inputs)
        x1 = reslist[1]
        x2 = reslist[2]
        x3 = reslist[3]
        x0 = self.upsample1(x3)
        x0 = torch.cat([x0, x2], 1)
        x0 = self.block1(x0)
        x0 = self.upsample0(x0)
        x0 = torch.cat([x0, x1], 1)
        x0 = self.block0(x0)
        return x0