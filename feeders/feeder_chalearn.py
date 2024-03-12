import sys
sys.path.extend(['../'])

import torch
import pickle
import numpy as np
from torch.utils.data import Dataset

from feeders import tools
import copy

class Feeder(Dataset):
    def __init__(self, data_path, label_path, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """
        :param data_path: the path for the preprocessed data, which is a .npz file
        :param label_path: the path for the preprocessed label, which is a .npz file
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel

        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        label_data = np.load(self.label_path)
        if self.split == 'train' or self.split == 'test_train':
            self.data = npz_data
            self.label = np.where(label_data > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
            print('data aug')
            N, T, _ = self.data.shape
            self.data = self.data.reshape((N, T, 1, 19, 3)).transpose(0, 4, 1, 3, 2)
            self.data, self.label = self.data_generation(self.data, self.label)
            # print(self.data.shape)
        elif self.split == 'test':
            self.data = npz_data
            self.label = np.where(label_data > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
            N, T, _ = self.data.shape
            self.data = self.data.reshape((N, T, 1, 19, 3)).transpose(0, 4, 1, 3, 2)
        else:
            raise NotImplementedError('data split only supports train/test')
        

    def data_generation(self, batch_datas, batch_labels):
        N, C, T, V, M = batch_datas.shape
        batch_datas = batch_datas.transpose(
            0, 4, 2, 3, 1).reshape(N * M, T, V * C)
        aug_train = copy.deepcopy(batch_datas)
        aug_train_label = copy.deepcopy(batch_labels)
        tmp_size = len(batch_datas)
        n_joints = batch_datas[0].shape[1]
        # generate augmented data
        for i in range(tmp_size):
            a = np.random.uniform(-np.pi / 36, np.pi / 36)
            b = np.random.uniform(-np.pi / 18, np.pi / 18)
            c = np.random.uniform(-np.pi / 36, np.pi / 36)
            tmpsample = np.zeros((1, 39, n_joints))
            for j in range(39):
                tmpmat = batch_datas[i][j].reshape(int(n_joints / 3), 3)
                tmpmat = tools.rotation(tmpmat, a, b, c)
                tmpsample[0][j] = tmpmat.reshape(n_joints)
            aug_train = np.concatenate((aug_train, tmpsample), axis=0)
        aug_train_label = np.concatenate(
            (aug_train_label, batch_labels), axis=0)

        tmp_train = np.zeros((tmp_size, 39 * n_joints))
        for i in range(tmp_size):
            tmp_train[i] = batch_datas[i].T.reshape(39 * n_joints)
        tmp_train = tools.translation(tmp_train, 5, 39)
        tmp_train2 = np.zeros((tmp_size, 39, n_joints))
        for i in range(tmp_size):
            tmp_train2[i] = tmp_train[i].reshape(n_joints, 39).T
        aug_train = np.concatenate((aug_train, tmp_train2), axis=0)
        aug_train_label = np.concatenate(
            (aug_train_label, batch_labels), axis=0)

        tmp_train3 = np.random.normal(
            0, 0.001, (tmp_size, 39, n_joints)) + batch_datas
        aug_train = np.concatenate((aug_train, tmp_train3), axis=0)
        aug_train_label = np.concatenate(
            (aug_train_label, batch_labels), axis=0)
        aug_train = aug_train.reshape(-1, M, T, V, C).transpose(0, 4, 2, 3, 1)

        return aug_train, aug_train_label

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(
            axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape(
            (N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]        
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy    

        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
            
        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod



