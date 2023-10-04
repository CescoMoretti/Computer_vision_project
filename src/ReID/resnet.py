import torch
import torch.nn as nn
from torchvision import models
from torchinfo import summary

# Model Resnet-50
class ft_net(nn.Module):
    def __init__(self, class_num = 751):
        super(ft_net, self).__init__()
        #Caricare il modello
        model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model = nn.Sequential(*list(model_ft.children())[:-1])
        self.model = model
        self.classifier = nn.Linear(2048, class_num, bias=False)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        score = self.classifier(x)
        return score
    
