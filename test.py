import os
from argparse import ArgumentError

import cv2
import numpy as np
import torch
import torch.utils.data as data_utils
from matplotlib import pyplot as plt

from config import Config
from datasets.mnist_dataset import MNIST_Dataset
from models.mnist_mil import Mnist_MIL_Net


config = Config()
experiment_dir = os.path.abspath(f'./results/{config.experiment_name}')
if not os.path.isdir(experiment_dir):
    raise Exception(f'Experiment directory {experiment_dir} does not exist.')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_dataset = MNIST_Dataset(
    bag_count=2048, mode='test', random_seed=config.dataset_random_seed)
test_loader = data_utils.DataLoader(test_dataset,
                                    batch_size=1,
                                    shuffle=False)

model = Mnist_MIL_Net()
model_path = os.path.join(experiment_dir, 'model.pt')
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

results_dir = os.path.join(experiment_dir, 'test/')
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
    
def draw_results(bag, label, save_dir):
    bag_device = bag.to(device)
    bag_device = bag_device.unsqueeze(1)
    with torch.no_grad():
        outputs = model(bag_device).cpu()
    print(outputs)
    rows = 2
    columns = 6
    fig = plt.figure(figsize=(12, 5))
    for i in range(1, len(bag) + 1):
        img = bag[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(bag[i - 1], cmap=plt.get_cmap('gray'))
        plt.title(f'Output: {outputs[i - 1].item():.2f}')

    plt.savefig(save_dir)

for index in range(10):
    bag, label = test_dataset[index]
    draw_results(bag, label, f'{results_dir}/example_{index}.png')

accuracy = 0
false_negative_example = None
false_positive_example = None
false_negative_count = 0
false_positive_count = 0
for bag, label in test_loader:
    bag_device = bag.to(device)
    bag_device = bag_device.transpose(0, 1)
    with torch.no_grad():
        outputs = model(bag_device).cpu()

    output = outputs.max()
    
    label_value = label.item()
    score = int(round(output.item()) == label_value)
    accuracy += score
    
    if score == 0:
        data = (bag_device.detach().cpu(), outputs.detach().cpu())
        if label_value == 1:
            false_negative_example = data
            false_negative_count += 1
        else:
            false_positive_example = data
            false_positive_count += 1
        
accuracy /= len(test_loader)

with open(f'{results_dir}/test_info.txt', mode='w') as f:
    text = f"""
Test information

test dataset size: {len(test_loader)}
test accuracy: {accuracy}

false positive count: {false_positive_count}
false negative count: {false_negative_count}
    """
    f.write(text)
    
    
if false_positive_example is not None:
    bag, label = false_positive_example
    bag = bag.squeeze(1)
    draw_results(bag, label, f'{results_dir}/false_positive_example.png')
    
if false_negative_example is not None:
    bag, label = false_negative_example
    bag = bag.squeeze(1)
    draw_results(bag, label, f'{results_dir}/false_negative_example.png')

print(f'Results with all statistics were saved to {results_dir}')
