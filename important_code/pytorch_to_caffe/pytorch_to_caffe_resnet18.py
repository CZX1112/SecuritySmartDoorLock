# -*- coding: utf-8 -*-
from __future__ import print_function  # do not delete this line if you want to save your log file.

import sys
sys.path.insert(0,'.')
import torch
from torch.autograd import Variable
from torchvision.models.resnet import *
# import pytorch_to_caffe
from pytorch_to_caffe_master import pytorch_to_caffe
from torchvision.models import resnet
import os
from naie.context import Context
import moxing as mox

if __name__=='__main__':
    name='resnet18'
    resnet18=resnet.resnet18(num_classes=5)
    checkpoint = torch.load("./latest.pth")
    new_sd = {}
    for k,v in checkpoint['state_dict'].items():
        if not k.endswith('num_batches_tracked'):
            if k.startswith('backbone'):
                k_new = k.split('backbone.')[1]
            if k.startswith('head'):
                k_new = k.split('head.')[1]
            new_sd[k_new] = v

    resnet18.load_state_dict(new_sd)
    resnet18.eval()
    input=torch.ones([1,3,224,224])
     #input=torch.ones([1,3,224,224])
    pytorch_to_caffe.trans_net(resnet18,input,name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    os.path.abspath('.')
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))
    os.path.abspath('.')
    mox.file.copy('resnet18.prototxt', os.path.join(Context.get_model_path(), 'resnet18.prototxt'))
    mox.file.copy('resnet18.caffemodel', os.path.join(Context.get_model_path(), 'resnet18.caffemodel'))
