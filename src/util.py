import torch
from torch.nn import BatchNorm2d, Conv2d, Linear
from torch.nn.modules.container import Sequential
from torchvision.models.resnet import BasicBlock

def rand_layers(model, module_paths):
    for module_path in module_paths:
        cur = model
        for name in module_path:
            cur = getattr(cur, name)
        randomize(cur)


def randomize(layer):
    if isinstance(layer, (Conv2d, Linear, BatchNorm2d)):
        # use previous statistical values for the randomization of that specific layer
        std, mean = torch.std_mean(layer.weight)
        layer.weight = torch.nn.Parameter(torch.empty(layer.weight.size()).normal_(mean=mean.item(),std=std.item()))
    elif isinstance(layer, (BasicBlock, Sequential)):
        for child in layer.children():
            randomize(child)