import os
import shutil

import torch
import torch.utils.data as data_utils
from matplotlib import pyplot as plt
from tqdm import tqdm

from config import Config
from datasets.mnist_dataset import MNIST_Dataset
from models.mnist_mil import Mnist_MIL_Net


config = Config()
experiment_dir = os.path.abspath(f'./results/{config.experiment_name}')
if not os.path.isdir(experiment_dir):
    os.mkdir(experiment_dir)
shutil.copyfile('./config.py', f'{experiment_dir}/config.py')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = MNIST_Dataset(bag_count=config.train_bag_count,
                        mode='train', random_seed=config.dataset_random_seed)
train_loader = data_utils.DataLoader(dataset,
                                     batch_size=1,
                                     shuffle=True)

val_dataset = MNIST_Dataset(bag_count=config.val_bag_count,
                            mode='val', random_seed=config.dataset_random_seed)
val_loader = data_utils.DataLoader(val_dataset,
                                   batch_size=1,
                                   shuffle=False)

model = Mnist_MIL_Net()
model = model.to(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=config.learning_rate, betas=config.adam_betas)


def loss_func(y, y_pred):
    return (y - y_pred) ** 2


pos = 0
neg = 0
for _, label in train_loader:
    label = label.item()
    if label == 1:
        pos += 1
    else:
        neg += 1
print(
    f'The training set has {pos} positive examples and {neg} negative examples.')

val_pos = 0
val_neg = 0
for _, label in val_loader:
    label = label.item()
    if label == 1:
        val_pos += 1
    else:
        val_neg += 1
print(
    f'The validation set has {val_pos} positive examples and {val_neg} negative examples.')

with open(f'{experiment_dir}/dataset_info.txt', mode='w') as f:
    text = f"""
    Dataset information
    
    positive training examples: {pos}
    negative training examples: {neg}
    
    positive validation examples: {val_pos}
    negative validation examples: {val_neg}
    """
    f.write(text)

loss_list = []
val_loss_list = []
acc_list = []
val_acc_list = []

for epoch_index in range(config.epoch_count):
    epoch_loss = 0
    epoch_validation_loss = 0

    epoch_acc = 0
    epoch_val_acc = 0

    model.train()
    loss = 0
    last_count = 0
    loader_len = len(train_loader)
    for count, (bag, label) in enumerate(tqdm(train_loader), start=1):
        bag = bag.transpose(0, 1)  # 1, bag_size, 28, 28 -> bag_size, 1, 28, 28
        bag = bag.to(device)
        label = label.to(device)

        output = model(bag).max()

        epoch_acc += int(round(output.item()) == label.item())
        loss += loss_func(label, output)

        if (count % config.batch_size == 0) or (count == loader_len):
            epoch_loss += loss.item() / (count - last_count)
            last_count = count
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = 0

    model.eval()
    with torch.no_grad():
        for bag, label in val_loader:
            bag = bag.transpose(0, 1)
            bag = bag.to(device)
            label = label.to(device)

            output = model(bag).max()

            epoch_val_acc += int(round(output.item()) == label.item())
            loss = loss_func(label, output)
            epoch_validation_loss += loss.item()

    epoch_validation_loss /= len(val_loader)

    epoch_acc /= len(train_loader)
    epoch_val_acc /= len(val_loader)

    loss_list.append(epoch_loss)
    val_loss_list.append(epoch_validation_loss)
    acc_list.append(epoch_acc)
    val_acc_list.append(epoch_val_acc)

    print(f'Epoch: {epoch_index}, train_loss: {epoch_loss:.4f}, val_loss: {epoch_validation_loss:.4f}, train_acc: {epoch_acc:.4f}, val_acc: {epoch_val_acc:.4f}')


torch.save(model.state_dict(), f'{experiment_dir}/model.pt')

plt.plot(loss_list)
plt.savefig(f'{experiment_dir}/loss.png')

plt.cla()
plt.plot(val_loss_list)
plt.savefig(f'{experiment_dir}/val_loss.png')

plt.cla()
plt.plot(acc_list)
plt.savefig(f'{experiment_dir}/acc.png')

plt.cla()
plt.plot(val_acc_list)
plt.savefig(f'{experiment_dir}/val_acc.png')
