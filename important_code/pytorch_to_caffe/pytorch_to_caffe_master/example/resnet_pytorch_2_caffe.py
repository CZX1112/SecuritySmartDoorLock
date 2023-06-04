import sys
sys.path.insert(0,'.')
import torch
from torch.autograd import Variable
from torchvision.models import resnet
import pytorch_to_caffe

if __name__=='__main__':
    name='resnet18'
    resnet18=resnet.resnet18(num_classes=5)
    checkpoint = torch.load("/home/ma-user/work/Algorithm/algo-pytorch_to_caffe_resnet18/latest.pth")
    new_sd = {}
    for k,v in checkpoint['state_dict'].items():
        if not k.endwith('num_batches_tracked'):
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
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))