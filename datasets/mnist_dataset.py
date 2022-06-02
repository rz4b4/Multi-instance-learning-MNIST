import numpy as np
import torch
import torch.utils.data as data_utils
import torchvision


class MNIST_Dataset(data_utils.Dataset):
    def __init__(self, mean_bag_size=10, std_bag_size=2, bag_count=1024, data_cache_dir='./datasets/cache/') -> None:
        self.mean_bag_size = mean_bag_size
        self.std_bag_size = std_bag_size
        self.bag_count = bag_count
        self.mnist_data = torchvision.datasets.MNIST(data_cache_dir, download=True)
        self.mnist_data_idxs = range(len(self.mnist_data))
        self.focus_label = 7
        
        self.bags, self.labels = self.generate_bags()
        
    def generate_bags(self) -> list:
        bags = []
        labels = []
        
        bag_lengths = np.random.normal(self.mean_bag_size, self.std_bag_size, size=(self.bag_count,)).astype(int)
        bag_lengths = bag_lengths.clip(5, 250000000)
        
        for _, bag_len in zip(range(self.bag_count), bag_lengths):
            indexes = np.random.choice(self.mnist_data_idxs, size=bag_len)
            imgs = self.mnist_data.data[indexes].type(torch.float) / 255
            img_labels = self.mnist_data.targets[indexes]
            
            bag_label = int(torch.any(img_labels == self.focus_label))
            
            bags.append(imgs)
            labels.append(bag_label)
        
        return bags, labels
    
    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, index):
        bag = self.bags[index]
        label = self.labels[index]
        return bag, label

if __name__ == '__main__':
    dataset = MNIST_Dataset()
