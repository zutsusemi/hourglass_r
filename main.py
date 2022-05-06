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
val_loader = DataLoader(val, 1, shuffle=True)
model = HourGlassNetwork(4, 14, 3, 256, 4).to(device)
# model = DeepPose(14)
model.to(device)
loss = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)




class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, loss, epoch, device, *args, **kwargs):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss = loss
        self.epoch = epoch
        self.save_checkpoint = kwargs['save_checkpoint_or_not'][0]
        self.num_iter_save = kwargs['save_checkpoint_or_not'][1]
        self.save_path = kwargs['save_path']
        self.load_checkpoint = kwargs['load_checkpoint_or_not']
        self.load_path = kwargs['path']
        self.max_r = kwargs['max_radius']
        if self.load_checkpoint == True:
            pth = self._load_checkpoint(self.load_path)
            self.model.load_state_dict(pth)
    
    def _loss(self, pred, anno):
        loss = 0
        for key, value in pred.items():
            loss += self.loss(value.float(), anno.float())
        return loss
    def _save_checkpoint(self, state_dict, path, num_iter):
        try:
            os.makedirs(path, exist_ok = True)
            torch.save(state_dict, path + num_iter + ".pth")
            print('[logger] checkpoint '+num_iter+' saved.')
        except Exception as e:
            print("error {}.".format(e))
    def _load_checkpoint(self, path):
        p = None
        try:
            p = torch.load(path)
            print('[logger] parameters loaded.')
        except Exception as e:
            print("error {}.".format(e))
        return p
    def train(self):
        self.model.train()
        # loss_value = float(0)
        count = 0
        for epoch in range(self.epoch):
            for j, [images, size, labels] in enumerate(self.train_loader):
                images, labels = images.to(device), labels.to(device)
                prediction = self.model(images)
                loss_value = self._loss(prediction, labels[:, :,  0 : 2].flatten(1).float()).float()
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
                count += 1
                if j % 10 == 0:
                    print('[epoch '+str(epoch)+' '+str(count)+'th iter]'+' loss: '+str(loss_value))
                if self.save_checkpoint == True and count % self.num_iter_save == 0:
                    self._save_checkpoint(self.model.state_dict(), self.save_path, str(count))
                    self.evaluate()
    def evaluate(self):
        self.model.eval()
        accuracy_list = []
        for j, [images, size, labels] in enumerate(self.val_loader):
            images, size, labels = images.to(device), size.to(device), labels.to(device)
            prediction = self.model(images)['result'] # (batch_size, 28)
            assert prediction.shape[0] == 1
            pred = prediction.reshape(prediction.shape[0], -1, 2)
            accuracy = self._accuracy(pred, labels, size)
            accuracy_list.append(accuracy)
        
        avg_accuracy = sum(accuracy_list) / len(accuracy_list)
        
        return avg_accuracy
    
    def _accuracy(self, pred, label, size):
        pred, label = pred.squeeze(), label.squeeze()
        size = size.squeeze()
        pred[:, 0] *= size[0]
        pred[:, 1] *= size[1]
        label[:, 0] *= size[0]
        label[:, 1] *= size[1]
        dist = ((pred - label[:,0 : 2]) ** 2).sum(axis = -1)
        right = dist < self.max_r
        accuracy = right.sum() / right.shape[0]
        return accuracy




# setup = {'save_checkpoint_or_not' : (1, 100),
#             'save_path': './checkpoints/',
#             'load_checkpoint_or_not': 0,
#             'path': None,
#             'max_radius': 20, }
trainer = Trainer(model, train_loader, val_loader, optimizer, loss, num_epochs, device, save_checkpoint_or_not=(1, 10), save_path="./checkpoints/", load_checkpoint_or_not=300, path=None, max_radius=20)  
trainer.train()





