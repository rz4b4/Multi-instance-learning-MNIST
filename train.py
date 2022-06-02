import cv2
import numpy as np
import torch
import torch.utils.data as data_utils
from tqdm import tqdm
from matplotlib import pyplot as plt

from config import Config
from datasets.mnist_dataset import MNIST_Dataset
from models.mnist_mil import Mnist_MIL_Net


config = Config()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = MNIST_Dataset(bag_count=128)
train_loader = data_utils.DataLoader(dataset,
                                     batch_size=1,
                                     shuffle=True)

test_dataset = MNIST_Dataset(bag_count=16)

model = Mnist_MIL_Net()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.adam_betas)

def loss_func(y, y_pred):
    return (y - y_pred) ** 2

for epoch_index in range(config.epoch_count):
    epoch_loss = 0
    tbar = tqdm(train_loader)
    for bag, label in tbar:
        bag = bag.transpose(0, 1) # 1, bag_size, 28, 28 -> bag_size, 1, 28, 28
        bag = bag.to(device)
        label = label.to(device)
        
        output = model(bag).max()
        
        loss = loss_func(label, output)
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
    
    print(f'Epoch: {epoch_index}, train loss: {epoch_loss:.4f}')


for index in range(16):
    bag, label = test_dataset[index]
    bag_device = bag.to(device)
    bag_device = bag_device.unsqueeze(1)
    print(bag_device.shape)
    outputs = model(bag_device).cpu()
    print(outputs)
    w = 10
    h = 10
    rows = 5
    columns = 5
    fig = plt.figure(figsize=(16, 16))
    for i in range(1, len(bag) + 1):
        img = np.random.randint(10, size=(h,w))
        fig.add_subplot(rows, columns, i)
        plt.imshow(bag[i - 1], cmap=plt.get_cmap('gray'))
        plt.title(f'Output: {outputs[i - 1].item():.2f}')
    plt.savefig(f'./results/example_{index}.png')
