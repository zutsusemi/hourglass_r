import torch
import torch.nn.functional as F
import numpy as np
import os
import PIL.Image as Image
from torch.utils.data import DataLoader
import torchvision.transforms as t
from data.preprocess import KeyptDataset
from model.hourglass_network import HourGlass, HourGlassNetwork
from model.deeppose import DeepPose

from transforms.transforms import Resize_Keypt, ToTensorKey
from tool.sampler import DatasetSampler
from util.normalize import normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device "+str(device)+".")

image = './lsp_dataset/images'
label = './lsp_dataset/joints.mat'

batchsize = 8
num_epochs = 5
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
# model = DeepPose(14)
model.to(device)
loss = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)




class Trainer:
    def __init__(self, model, train_loader, optimizer, loss, epoch, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.loss = loss
        self.epoch = epoch
    
    def _loss(self, pred, anno):
        loss = 0
        for key, value in pred.items():
            loss += self.loss(value.float(), anno.float())
        return loss
    def train(self):
        model.train()
        # loss_value = float(0)
        count = 0
        for epoch in range(self.epoch):
            for j, [images, size, labels] in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                prediction = model(images)
                loss_value = self._loss(prediction, labels[:, :,  0 : 2].flatten(1).float()).float()
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
                count += 1
                if j % 10 == 0:
                    print('[epoch '+str(epoch)+' '+str(count)+'th iter]'+' loss: '+str(loss_value))


trainer = Trainer(model, train_loader, optimizer, loss, num_epochs, device)
trainer.train()





