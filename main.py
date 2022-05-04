import torch
import torch.nn.functional as F
import numpy as np
import os
import PIL.Image as Image
from torch.utils.data import DataLoader
import torchvision.transforms as t
from data.preprocess import KeyptDataset
from model.hourglass_network import HourGlass, HourGlassNetwork

from transforms.transforms import Resize_Keypt, ToTensorKey
from tool.sampler import DatasetSampler
from util.normalize import normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device "+str(device)+".")

image = './lsp_dataset/images'
label = './lsp_dataset/joints.mat'

batchsize = 8
num_epochs = 1
img_size = 256
lr = 0.00001

transforms = t.Compose([ToTensorKey(),
                            Resize_Keypt((img_size, img_size))])
dataset = KeyptDataset(transforms, image, label)
sampler = DatasetSampler(len(dataset), len(dataset) // 5)
train, val = sampler(dataset)
train_loader = DataLoader(train, batchsize, shuffle=True)
val_loader = DataLoader(val, batchsize, shuffle=True)
model = HourGlassNetwork(4, 14, 3, 256, 4).to(device)
loss = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


model.train()
# loss_value = float(0)

for j, [images, size, labels] in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)
    prediction = model(images)
    loss_value = loss(prediction['result'].float(), labels[:, :,  0 : 2].float()).float()
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    if j % 10 == 0:
        print('['+str(j)+'th iter]'+' loss: '+str(loss_value))





