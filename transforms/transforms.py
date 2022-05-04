import torch
from torchvision.transforms import *

class Resize_Keypt(torch.nn.Module):
    def __init__(self, size, interpolation = InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        self.Resize = Resize(size)
        self.size = size
    def forward(self, a):
        '''
        anno: (14, 3)
        '''
        image = a['image']
        anno = a['anno']
        new_image = self.Resize(image)
        H, W = image.shape[-2], image.shape[-1]
        anno[:, 0] = anno[:, 0] / W
        anno[:, 1] = anno[:, 1] / H
        return {'image': new_image, 'anno': anno}

class ToTensorKey(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.totensor = ToTensor()
    def forward(self, a):
        '''
        anno: (14, 3)
        '''
        new_image = self.totensor(a['image'])
        anno = self.totensor(a['anno']).squeeze()
        return {'image': new_image, 'anno': anno}




