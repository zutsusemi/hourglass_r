
from sklearn.preprocessing import scale
import torch
import torch.nn.functional as F
import numpy as np

class BasicBlock(torch.nn.Module):
    def __init__(self,  
                in_channels=128, 
                out_channels=128,
                size=1, 
                stride=1,
                padding=0,
                norm='bn'):
        super().__init__()
        self.in_channels = in_channels
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, size, stride, padding)
    def forward(self, x):
        return self.conv(F.relu(self.bn(x)))

class ResidualMod(torch.nn.Module):
    def __init__(self, 
                layers=3, 
                in_channels=[128, 128, 128],
                out_channels=[128, 128, 256], 
                stride=[1, 1, 2],
                size=[1, 3, 1],
                padding=[0, 1, 0],
                norm='bn'):
        super().__init__()
        self.ResidualMod = self._make_layers(layers, in_channels, out_channels, stride, size, padding, norm)
        self.ResidualConnect = torch.nn.Conv2d(out_channels[-2], out_channels[-1], 1, 1) if out_channels[-2] != out_channels[-1] else None
    
    def _make_layers(self, layers, in_channels, out_channels, stride, size, padding, norm):
        module_list = [BasicBlock(in_channels[j], out_channels[j], size[j], stride[j], padding[j], norm) for j in range(layers)]
        return torch.nn.Sequential(*module_list)
    
    def forward(self, x):
        out = self.ResidualMod(x)
        if self.ResidualConnect == None:
            return out + x
        x = self.ResidualConnect(x)
        return out + x

class HourGlass(torch.nn.Module):
    def __init__(self, order=4):
        super().__init__()
        self.hg_module = self._make_hg(order)
        self.order = order
    def _make_hg(self, order=4):
        module_list = []
        for j in range(order):
            for m in range(3):
                module_list.append(ResidualMod(3, [256, 256, 256], [256, 256, 256], [1, 1, 1]))
        module_list.append(ResidualMod(3, [256, 256, 256], [256, 256, 256], [1, 1, 1]))
        return torch.nn.ModuleList(module_list)
    def forward(self, x):
        out = self._forward_helper(x, 0, self.order)
        return out
    def _forward_helper(self, x, count, order=4):
        out = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        out = self.hg_module[3*count](out)
        out_branch = self.hg_module[3*count+2](out)
        if count < order - 1:
            out = self._forward_helper(out, count + 1, order)
        else:
            out = self.hg_module[3*count+3](out)
        out = out + out_branch
        out = F.interpolate(self.hg_module[3*count+1](out), scale_factor=2)
        return out + x

class HourGlassNetwork(torch.nn.Module):
    def __init__(self, num_hg=4, num_classes=14, in_channel=128, input_dim=256, order=4, input_size=256*256):
        super().__init__()
        self.num_hg = num_hg
        self.start = self._make_start(in_channel, input_dim)
        self.hg_net = self._make_hg(num_hg, order, [input_dim for _ in range(num_hg + 1)], num_classes)
        self.fc = torch.nn.ModuleList([torch.nn.Linear(input_size // (4*4), 2 * num_classes) for _ in range(num_hg)])
    
    def _make_start(self, in_channel, input_dim=256):
        seq = torch.nn.Sequential(torch.nn.Conv2d(in_channel, 64, kernel_size=3, stride=2, padding=1),
                                    torch.nn.BatchNorm2d(64),
                                    torch.nn.ReLU(),
                                    ResidualMod(3, [64, 64, 64], [64, 64, 128],[1, 1, 1]),
                                    torch.nn.MaxPool2d(3, 2, padding=1),
                                    ResidualMod(3, [128, 128, 128], [128, 128, 128], [1, 1, 1]),
                                    ResidualMod(3, [128, 128, 128], [128, 128, input_dim], [1, 1, 1])
        )
        return seq
    def _make_hg(self, num_hg, order=4, channels=[256, 256, 256, 256, 256], num_cls=1):
        module_list = []
        j = 0
        for j in range(num_hg - 1):
            module_list.append(HourGlass(order=order))
            module_list.append(ResidualMod(1, [channels[j] for _ in range(3)], [channels[j] for _ in range(3)], [1, 1, 1]))
            module_list.append(torch.nn.Conv2d(channels[j], 1, 1, bias=True))
            module_list.append(torch.nn.Conv2d(1, channels[j+1], 1, bias=True))
            module_list.append(torch.nn.Conv2d(channels[j], channels[j+1], 1, bias=True))
        
        module_list.append(HourGlass(order=order))
        module_list.append(torch.nn.Conv2d(channels[j+1], 1, 1))
        return torch.nn.ModuleList(module_list)
    
    def forward(self, img):
        x = self.start(img)
        result = dict()
        for j in range(self.num_hg - 1):
            out = self.hg_net[j*5](x)
            ll = self.hg_net[j*5+1](out)
            out = self.hg_net[j*5+4](ll)

            map = self.hg_net[j*5+2](ll)
            branch = self.hg_net[j*5+3](map)
            result['heat_map_'+str(j)]=self.fc[j](map.flatten(1)).sigmoid()
            x = out + branch + x 
        out = self.hg_net[-1](self.hg_net[-2](x))
        out = out.flatten(1)
        out = self.fc[j](out).sigmoid()
        result['result'] = out
        return result




# if __name__ == '__main__':
#     model = HourGlassNetwork(4, 14, 3, 256, 4)
#     img = torch.rand(2, 3, 256, 256)
#     out = model(img)
        

        



    
            


        




