import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from resnet import resnet50, resnet18
import torch.nn.functional as F
import math
from attention import IWPA, AVG, MAX, GEM

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out



# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)]
        feat_block += [nn.BatchNorm1d(low_dim)]

        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block

    def forward(self, x):
        x = self.feat_block(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]

        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        return x

class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        x = self.visible.layer1(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x = self.thermal.layer1(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        #x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class embed_net(nn.Module):
    def __init__(self, class_num, drop=0.2, part = 3, arch='resnet50', cpool = 'no', bpool = 'avg', fuse = 'sum'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        pool_dim = 2048
        pool_dim_att = 2048 if fuse == "sum" else 4096
        self.dropout = drop
        self.part = part
        self.cpool = cpool
        self.bpool = bpool
        self.fuse = fuse

        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        if self.cpool == 'wpa':
            self.classifier_att = nn.Linear(pool_dim_att, class_num, bias=False)    
            self.classifier_att.apply(weights_init_classifier)
            self.cpool_layer = IWPA(pool_dim, part,fuse)
        if self.cpool == 'avg':
            self.classifier_att = nn.Linear(pool_dim_att, class_num, bias=False)    
            self.classifier_att.apply(weights_init_classifier)
            self.cpool_layer = AVG(pool_dim,fuse)
        if self.cpool == 'max':
            self.classifier_att = nn.Linear(pool_dim_att, class_num, bias=False)    
            self.classifier_att.apply(weights_init_classifier)
            self.cpool_layer = MAX(pool_dim,fuse)
        if self.cpool == 'gem':
            self.classifier_att = nn.Linear(pool_dim_att, class_num, bias=False)    
            self.classifier_att.apply(weights_init_classifier)
            self.cpool_layer = GEM(pool_dim,fuse)



    def forward(self, x1, x2, modal=0):
        # domain specific block
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        # shared four blocks
        x = self.base_resnet(x)

        if self.bpool == 'gem':
            b, c, _, _ = x.shape
            x_pool = x.view(b, c, -1)
            p = 3.0    
            x_pool = (torch.mean(x_pool**p, dim=-1) + 1e-12)**(1/p)
        elif self.bpool == 'avg':
            x_pool = F.adaptive_avg_pool2d(x,1)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
        elif self.bpool == 'max':
            x_pool = F.adaptive_max_pool2d(x,1)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
        else:
            print("wrong backbone pooling!!!")
            exit()

        feat  = self.bottleneck(x_pool)

        if self.cpool != 'no':
            # intra-modality weighted part attention
            if self.cpool == 'wpa':
                feat_att, feat_att_bn = self.cpool_layer(x, feat, 1, self.part)
            if self.cpool in ['avg', 'max', 'gem']:
                feat_att, feat_att_bn = self.cpool_layer(x, feat)

            if self.training:            
                return x_pool, self.classifier(feat), feat_att_bn, self.classifier_att(feat_att_bn) 
            else:
                return self.l2norm(feat), self.l2norm(feat_att_bn)
        else:
            if self.training:            
                return x_pool, self.classifier(feat)
            else:
                return self.l2norm(feat)