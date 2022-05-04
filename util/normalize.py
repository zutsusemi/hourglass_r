import torch

def normalize(size, labels):
    labels[:, :, 0] = labels[:, :, 0] // size[:, 0].unsqueeze(1)
    labels[:, :, 1] = labels[:, :, 1] // size[:, 1].unsqueeze(1)
    return labels