######### STEP COMMIT
# 1) Sistemare la classificazione


import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable

######################################################################
### Function
### --------
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


#############################################à
### Model
### -----
class ReIDMOdel(nn.Module):
    def __init__(self, class_num=751, droprate = 0.5):
        super(ReIDMOdel, self).__init__()

        res = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = nn.Sequential(*list(res.children())[:-1]) # Without classifier
        final_dim = 2048
        add_block = []
        add_block += [nn.BatchNorm1d(final_dim)]
        #if droprate > 0:
        #    add_block  += [nn.Dropout(p = droprate)]
        self.hide = nn.Sequential(*add_block)
        self.hide.apply(weights_init_kaiming)

        self.classifier = nn.Linear(final_dim, class_num)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), x.size(1))
        x = self.hide(x)
        x = self.classifier(x)

        return x
        