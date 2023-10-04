import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from resnet import ft_net

h, w = 256, 128
data_dir = "C:/Users/gabri/Downloads/Market-1501-v15.09.15/pytorch"
batchsize = 1
num_epochs = 2
use_gpu = torch.cuda.is_available()

transforms_train_list = [
    transforms.Resize((h,w), interpolation=3),
    transforms.Pad(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transforms_var_list = [
    transforms.Resize((h,w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

data_trasforms = {
    'train': transforms.Compose(transforms_train_list),
    'val': transforms.Compose(transforms_var_list)
}

image_dataset = {}
image_dataset['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_trasforms['train'])
image_dataset['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_trasforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_dataset[x], batch_size=batchsize, shuffle=True)
               for x in ['train', 'val']}

class_names = image_dataset['train'].classes
model = ft_net(len(class_names))

criterion = nn.CrossEntropyLoss()
optim_name = optim.SGD

lr = 0.05

ignored_params = list(map(id, model.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
classifier_params = model.classifier.parameters()
optimizer = optim_name([
    {'params': base_params, 'lr': 0.1*lr},
    {'params': classifier_params, 'lr':lr}
], weight_decay=5e-4, momentum=0.9, nesterov=True)

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train(True)  # Set model to training mode
        else:
            model.train(False)  # Set model to evaluate mode

        # Iterate over data.
        for data in dataloaders[phase]:
            # get a batch of inputs
            inputs, labels = data
            now_batch_size, c, h, w = inputs.shape
            if now_batch_size < batchsize:  # skip the last batch
                continue
            # print(inputs.shape)
            # wrap them in Variable, if gpu is used, we transform the data to cuda.
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # -------- forward --------
            outputs = model(inputs)
            preds = torch.max(outputs.data, 0)
            loss = criterion(outputs, labels)

            # -------- backward + optimize --------
            # only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()