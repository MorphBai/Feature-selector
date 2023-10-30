from scipy.signal import butter, sosfilt
import pandas as pd
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torch
import os
import warnings
import utils.utils as utils
warnings.filterwarnings("ignore")


def prepare(experiment_name, task, datafolder, outputfolder,
            sampling_frequency=100, min_samples=300,    # sampling_frequency可以为100或500
            train_fold=8, val_fold=9, test_fold=10):    # 这个函数的作用是将数据集划分为训练集、验证集和测试集
    # Load PTB-XL data
    data, raw_labels = utils.load_dataset(datafolder, sampling_frequency)   # data是信号数据，raw_labels是标签数据
    # Preprocess label data
    labels = utils.compute_label_aggregations(raw_labels, datafolder, task) # 将标签数据转化为one-hot编码

    save_folder = os.path.join(outputfolder, experiment_name, 'data/')  # 保存数据的文件夹
    os.makedirs(save_folder, exist_ok=True) # 创建文件夹

    # Select relevant data and convert to one-hot
    data, labels, Y, _ = utils.select_data(
        data, labels, task, min_samples, save_folder)   # 选择数据

    # 10th fold for testing (9th for now)
    X_test = data[labels.strat_fold == test_fold]   # 选择测试集
    y_test = Y[labels.strat_fold == test_fold]  # 选择测试集标签
    # 9th fold for validation (8th for now)
    X_val = data[labels.strat_fold == val_fold] # 选择验证集
    y_val = Y[labels.strat_fold == val_fold]    # 选择验证集标签
    # rest for training
    X_train = data[labels.strat_fold <= train_fold]  # 选择训练集
    y_train = Y[labels.strat_fold <= train_fold]    # 选择训练集标签

    # Preprocess signal data
    X_train, X_val, X_test = utils.preprocess_signals(
        X_train, X_val, X_test, save_folder)    # 对信号数据进行预处理
    n_classes = y_train.shape[1]    # 计算类别数

    # save train and test labels
    y_train.dump(os.path.join(  # dump函数的作用是将数据保存为npy格式
        outputfolder, experiment_name, 'data/y_train.npy'))   # 保存训练集标签
    y_val.dump(os.path.join(outputfolder, experiment_name, 'data/y_val.npy'))   # 保存验证集标签
    y_test.dump(os.path.join(
        outputfolder, experiment_name, 'data/y_test.npy'))  # 保存测试集标签

    return (X_train, X_val, X_test), (y_train, y_val, y_test), n_classes    # 返回数据集、标签和类别数


class ECGData(Dataset): # 这个类的作用是将数据转化为torch需要的格式
    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = X
        self.y = y

    def __getitem__(self, index):
        data = torch.tensor(self.X[index], dtype=torch.float32) # 将数据转化为torch需要的格式
        data = data.transpose(0, 1)  # 将数据转化为torch需要的size
        label = torch.tensor(self.y[index], dtype=torch.float)  # 将标签转化为torch需要的格式
        return {'features': data, 'labels': label}  # 返回数据和标签

    def __len__(self):
        return self.X.shape[0]  # 返回数据集大小


class ECGDataLongformer(Dataset):   # 这个类的作用是将数据转化为longformer需要的格式
    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = X
        self.y = y

    def __getitem__(self, index):
        data = torch.tensor(self.X[index], dtype=torch.float32) # 将数据转化为torch需要的格式
        data = torch.cat([data, torch.zeros((24, 12), dtype=torch.long)])   # 将数据转化为longformer需要的格式
        mask = torch.tensor([1] * 1000 + [0] * 24, dtype=torch.long)
        # data = data.transpose(0, 1)  # 将数据转化为torch需要的size
        label = torch.tensor(self.y[index], dtype=torch.long)   # 将标签转化为torch需要的格式
        return {'features': data, 'attention_mask': mask, 'labels': label}  # 返回数据、mask和标签

    def __len__(self):
        return self.X.shape[0]  # 返回数据集大小


def create_dataloaders(args, train_mode=True):  # 这个函数的作用是创建dataloader
    X, y, n_classes = prepare(
        args.task, args.task, './dataset/ptb-xl/', 'experiements', sampling_frequency=100)  # 读取数据集
    X_train, X_val, X_test = X  # 读取数据集，X是数据，y是标签
    y_train, y_val, y_test = y

    if train_mode:  # 如果是训练模式
        train_datasets = ECGData(X_train, y_train)  # 创建训练集
        train_iter = DataLoader(train_datasets, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers, pin_memory=True)    # 创建训练集dataloader
        val_datasets = ECGData(X_val, y_val)    # 创建验证集
        val_iter = DataLoader(val_datasets, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)  # 创建验证集dataloader
        return train_iter, val_iter   # 返回训练集dataloader和验证集dataloader
    else:
        test_datasets = ECGData(X_test, y_test) # 创建测试集
        test_iter = DataLoader(test_datasets, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers, pin_memory=True)    # 创建测试集dataloader
        return test_iter


if __name__ == '__main__':
    X, y, n_classes = prepare(
        'exp1', 'all', './dataset/ptb-xl/', 'experiements') # 读取数据集
    X_train, X_val, X_test = X  # 读取数据集，X是数据，y是标签
    y_train, y_val, y_test = y

    data = ECGData(X_train, y_train)    # 创建数据集
    for x, y in data:
        print('X, y shape', x.shape, y.shape)   # 打印数据集大小
        break
