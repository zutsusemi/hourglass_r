from sklearn.preprocessing import scale
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

class DeepPose(torch.nn.Module):
	def __init__(self, nJoints, modelName='resnet34'):
		super(DeepPose, self).__init__()
		self.nJoints = nJoints
		self.block = 'BottleNeck' if (int(modelName[6:]) > 34) else 'BasicBlock'
		self.resnet = getattr(models, modelName)(pretrained=False)
		self.resnet.fc = torch.nn.Linear(512 * (4 if self.block == 'BottleNeck' else 1), self.nJoints * 2)
	def forward(self, x):
		return self.resnet(x).sigmoid()
