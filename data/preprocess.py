from matplotlib import image
import torch
from torch.utils import data
import os
import scipy.io
import numpy as np
from PIL import Image

class KeyptDataset(data.Dataset):
    def __init__(self, transforms, image_pth, ann_pth):
        self.image_list = sorted(os.listdir(image_pth))
        self.transforms = transforms
        self.ann_mat = scipy.io.loadmat(ann_pth)
        joints = self.ann_mat["joints"]
        joints = np.swapaxes(joints, 0, 2)
        # structure of dataset:
        '''
        [{'image': , 'original_size': , 'anno':, }]
        '''
        
        self.dataset = []

        for file_id, file_name in enumerate(self.image_list):
            this_data = dict()
            image = Image.open(os.path.join(image_pth, file_name))
            this_data['image'] = image
            this_data['original_size'] = image.size
            this_data['anno'] = joints[file_id].squeeze()
            self.dataset.append(this_data)
    
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        out = self.transforms(self.dataset[index])
        this_size = torch.tensor(self.dataset[index]['original_size'])
        return out['image'], this_size, out['anno']
    
        


