import numpy as np
import torch
import torch.utils.data as data_utils
import torchvision


class MNIST_Dataset(data_utils.Dataset):
    def __init__(self, mean_bag_size=10, std_bag_size=2, bag_count=1024, data_cache_dir='./datasets/cache/', mode='train') -> None:
        self.mean_bag_size = mean_bag_size
        self.std_bag_size = std_bag_size
        self.bag_count = bag_count
        self.mnist_data = torchvision.datasets.MNIST(data_cache_dir, download=True)
        
        assert mode in ['train', 'test', 'val'], "Dataset mode not supported. Available options: 'train', 'test', 'val'."
        
        if mode == 'train':
            self.data_min_idx = 0
            self.data_max_idx = int(len(self.mnist_data) * 0.80) - 1
        elif mode == 'val':
            self.data_min_idx = int(len(self.mnist_data) * 0.80)
            self.data_max_idx = int(len(self.mnist_data) * 0.9) - 1
        else:
            self.data_min_idx = int(len(self.mnist_data) * 0.9)
            self.data_max_idx = len(self.mnist_data) - 1
            
        self.focus_label = 7
        
        self.positive_datapoints, self.negative_datapoints = self.generate_datapoints()
        self.bags, self.labels = self.generate_balanced_dataset()
        
    def generate_datapoints(self) -> list:
        bags = []
        labels = []
        
        bag_lengths = np.random.normal(self.mean_bag_size, self.std_bag_size, size=(self.bag_count,)).astype(int)
        bag_lengths = bag_lengths.clip(5, 250000000)
        
        for _, bag_len in zip(range(self.bag_count), bag_lengths):
            indexes = np.random.randint(self.data_min_idx, self.data_max_idx, size=bag_len)
            imgs = self.mnist_data.data[indexes].type(torch.float) / 255
            img_labels = self.mnist_data.targets[indexes]
            
            bag_label = int(torch.any(img_labels == self.focus_label))
            
            bags.append(imgs)
            labels.append(bag_label)
            
        positive_datapoints = [(bag, label) for bag, label in zip(bags, labels) if label == 1]
        negative_datapoints = [(bag, label) for bag, label in zip(bags, labels) if label == 0]
        
        return positive_datapoints, negative_datapoints
    
    def generate_balanced_dataset(self):
        bags = []
        labels = []
        for pos, neg in zip(self.positive_datapoints, self.negative_datapoints):
            bags.append(pos[0])
            labels.append(pos[1])
            bags.append(neg[0])
            labels.append(neg[1])
        return bags, labels
    
    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, index):
        bag = self.bags[index]
        label = self.labels[index]
        return bag, label

if __name__ == '__main__':
    dataset = MNIST_Dataset()
