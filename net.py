import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.models.feature_extraction import create_feature_extractor
import math
from resnet import ResNet50


class FeatureCoupling(nn.Module):
    def __init__(self, cfilter, tfilter):
        super().__init__()
        self.conv_conv = nn.Conv2d(tfilter, cfilter, 1, stride=1, padding= 'same')
        self.conv_trans = nn.Conv2d(cfilter, tfilter, 1, stride=1, padding= 'same')
        self.batch_conv = nn.BatchNorm2d(cfilter)
        self.batch_trans = nn.BatchNorm2d(tfilter)

    def forward(self, inputs_conv, inputs_trans):
        #inputs_conv, inputs_trans = inputs
        b, cc, whc = inputs_conv.shape
        b, ct, wt, ht = inputs_trans.shape
        inputs_conv = torch.reshape(inputs_conv, [-1, cc, int(math.sqrt(whc)), int(math.sqrt(whc))])
        inputs_conv = self.conv_conv(inputs_conv)
        inputs_conv = self.batch_conv(inputs_conv)
        inputs_conv = torch.nn.functional.gelu(inputs_conv)
        #inputs_conv = F.interpolate(inputs_conv, size=[wt, ht], mode='bilinear')

        inputs_trans = self.conv_trans(inputs_trans)
        inputs_trans = self.batch_trans(inputs_trans)
        inputs_trans = torch.nn.functional.gelu(inputs_trans)
        #inputs_trans = F.interpolate(inputs_trans, size=[int(math.sqrt(whc)), int(math.sqrt(whc))], mode='bilinear')
        inputs_trans = torch.reshape(inputs_trans, [-1, cc, whc])
        
        return inputs_conv, inputs_trans

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

class CFAConv(nn.Module):
    def __init__(self, cfilter):
        super().__init__()
        self.conv0 = nn.Conv2d(cfilter, cfilter//2, 1, stride=1)
        self.conv1 = nn.Conv2d(cfilter, cfilter//2, 1, stride=1)
        self.conv2 = nn.Conv2d(cfilter, cfilter//2, 1, stride=1)
        self.conv3 = nn.Conv2d(cfilter//2, cfilter, 1, stride=1)
        self.gap = nn.AdaptiveMaxPool2d((1, 1))
        self.conv4 = conv_block(cfilter*2, cfilter)
        #self.dense0 = nn.Linear(cfilter, 1000)
        self.batch_conv0 = nn.BatchNorm2d(cfilter)

    def forward(self, inputs_0, inputs_1):
        x01 = self.conv0(inputs_0)
        x2 = self.conv1(inputs_0)
        x3 = self.conv2(inputs_1)
        b, c, w, h = x3.shape
        x1 = x01.view(-1, c, w * h)
        x2 = x2.view(-1, c, w * h)
        x3 = x3.view(-1, c, w * h)
        x0 = torch.einsum('bij,bik->bjk', x2, x3)
        x0 = F.softmax(x0, dim=-2)
        x0 = torch.einsum('bij,bjk->bik', x1, x0)
        x0 = x0.view(-1, c, w, h)
        x0 = self.conv3(x0)
        x0 = self.batch_conv0(x0)
        x0 = torch.nn.functional.gelu(x0)
        output = self.conv4(torch.cat([x0,inputs_0], 1))
        #output = torch.squeeze(self.gap(torch.nn.functional.gelu(x0+inputs_0)), [2, 3])
        output = torch.squeeze(self.gap(output+x0), 2)
        output = torch.squeeze(output, 2)
        #return torch.nn.functional.gelu(self.dense0(output))
        return output

class conv_block2(nn.Module):
    def __init__(self, infilter, outfilter):
        super().__init__()
        self.conv0 = nn.Conv1d(infilter, outfilter, 3, stride=1, padding='same')
        self.conv1 = nn.Conv1d(outfilter, outfilter, 3, stride=1, padding='same')
        self.batch_conv0 = nn.BatchNorm1d(outfilter)
        self.batch_conv1 = nn.BatchNorm1d(outfilter)

    def forward(self, inputs):
        x0 = self.conv0(inputs)
        x0 = self.batch_conv0(x0)
        x0 = torch.nn.functional.gelu(x0)
        x0 = self.conv1(x0)
        x0 = self.batch_conv1(x0)
        x0 = torch.nn.functional.gelu(x0)
        return x0

class CFATrans(nn.Module):
    def __init__(self,tfilter):
        super().__init__()
        self.conv0 = nn.Conv1d(tfilter, tfilter//2, 1, stride=1)
        self.conv1 = nn.Conv1d(tfilter, tfilter//2, 1, stride=1)
        self.conv2 = nn.Conv1d(tfilter, tfilter//2, 1, stride=1)
        self.conv3 = nn.Conv1d(tfilter//2, tfilter, 1, stride=1)
        self.gap = nn.AdaptiveMaxPool1d(1)
        self.conv4 = conv_block2(tfilter*2, tfilter)
        #self.dense0 = nn.Linear(tfilter, 1000)
        self.batch_conv0 = nn.BatchNorm1d(tfilter)

    def forward(self, inputs_0, inputs_1):
        x1 = self.conv0(inputs_0)
        x2 = self.conv1(inputs_0)
        x3 = self.conv2(inputs_1)
        x0 = torch.einsum('bij,bik->bjk', x2, x3)
        x0 = F.softmax(x0, dim=-2)
        x0 = torch.einsum('bij,bjk->bik', x1, x0)
        x0 = self.conv3(x0)
        x0 = self.batch_conv0(x0)
        x0 = torch.nn.functional.gelu(x0)
        output = self.conv4(torch.cat([x0,inputs_0], 1))
        output = torch.squeeze(self.gap(output+x0), 2)
        #return torch.nn.functional.gelu(self.dense0(output))
        return output


class CrossViT(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        cfilter = 256
        tfilter = 768
        self.vit = create_feature_extractor(
            timm.create_model('vit_base_patch32_384', pretrained=pretrained)
            , return_nodes=['blocks.11'])
        self.resnet50 = ResNet50(cfilter, pretrained=pretrained)
        self.FeatureC = FeatureCoupling(cfilter = cfilter, tfilter = tfilter)
        self.CFAC = CFAConv(cfilter = cfilter)
        self.CFAT = CFATrans(tfilter = tfilter)
        self.denseC = nn.Linear(1000, 1)
        self.denseT = nn.Linear(1000, 1)
        self.dense0 = nn.Linear(tfilter, 1000)
        self.dense1 = nn.Linear(cfilter, 1000)

    def forward(self, inputs):
        inputs1 = F.interpolate(inputs, size=[384, 384], mode='bilinear')
        x1 = self.vit(inputs1)['blocks.11'][:,1:,:].transpose(1,2)
        x2 = self.resnet50(inputs)
        xc1, xc2 = self.FeatureC(x1, x2)
        xo1 = self.CFAC(x2, xc1)
        xo2 = self.CFAT(x1, xc2)
        xo1 = torch.nn.functional.gelu(self.dense1(xo1))
        xo2 = torch.nn.functional.gelu(self.dense0(xo2))
        xo1 = F.sigmoid(self.denseC(xo1))
        xo2 = F.sigmoid(self.denseT(xo2))
        #return torch.squeeze((xo1 + xo2) / 2, -1)
        return torch.squeeze(xo1, -1), torch.squeeze(xo2, -1)