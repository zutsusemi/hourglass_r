from matplotlib import image
import matplotlib.pyplot as plt
import torch
from torch.utils import data
import os
import scipy.io
import numpy as np
from PIL import Image
import torchvision.transforms as t
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

class Visualizer:
    def __init__(self, image_pth, anno_pth):
        self.image_list = sorted(os.listdir(image_pth))
        self.ann_mat = scipy.io.loadmat(anno_pth)
        joints = self.ann_mat["joints"]
        joints = np.swapaxes(joints, 0, 2)
        self.list = []
        self.t = t.ToTensor()
        for file_id, file_name in enumerate(self.image_list):
            this_data = dict()
            image = Image.open(os.path.join(image_pth, file_name))
            this_data['image'] = self.t(image)
            this_data['original_size'] = image.size
            this_data['anno'] = joints[file_id].squeeze()
            self.list.append(this_data)
    def visualize(self, file_id):
        image = self.list[file_id]['image'].permute(1,2,0)
        anno = self.list[file_id]['anno']
        plt.figure()
        plt.imshow(image)
        plt.scatter(anno[:, 0], anno[:, 1], color='yellow')



if __name__ == '__main__':
    visual = Visualizer('./lsp_dataset/images', './lsp_dataset/joints.mat')
    visual.visualize(0)
    visual.visualize(110)

