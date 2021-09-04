import logging
from collections import OrderedDict
import torch
from models.ResNet import resnet18
import torch.nn.functional as F


def nonadaptiveconcatpool2d(x, k):
    # concatenating average and max pool, with kernel and stride the same
    ap = F.avg_pool2d(x, kernel_size=(k, k), stride=(k, k))
    mp = F.max_pool2d(x, kernel_size=(k, k), stride=(k, k))
    return torch.cat([mp, ap], 1)

class Network(torch.nn.Module):
    def __init__(self, architecture='ResNet', n_classes=2, pretrained=True):
        super().__init__()

        if architecture == 'ResNet':
            model = resnet18(pretrained=pretrained)
            self.backbone = torch.nn.Sequential(*list(model.children())[:-2])
            num_features = 512*2

        self.dropout = torch.nn.Dropout(p=0.8)
        self.fc1 = torch.nn.Linear(num_features, n_classes)

        self.selected_out = OrderedDict()
        self.fhooks = []
        #hook_layer_names = ['fc1', 'backbone.7.1.conv2']
        hook_layer_names = ['fc1']
        for name, layer in self.named_modules():
            if name in hook_layer_names:
                self.fhooks.append(layer.register_forward_hook(self.forward_hook(name)))


    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook


    def forward(self, x):
        x = self.backbone(x)
        kernel_size = x.shape[-1] 
        x = nonadaptiveconcatpool2d(x, kernel_size)
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)        
        return x

        

    