import sys
sys.path.extend(['../'])

import torch
import pickle
import cv2
import numpy as np
from torch.utils.data import Dataset

from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False,  random_miss=False):
        """
        :param data_path: the path for the preprocessed data, which is a .npz file
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param random_miss: If true, randomly drop some frames, only set true for robustness test at test phrase after training
        """

        self.debug = debug
        self.data_path = data_path
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
        self.random_miss = random_miss

        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        # print(self.data.shape)
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)


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
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy    
        else: 
            #From the code of Hyperformer
            # # there's a freedom to choose the direction of local coordinate axes!
            trajectory = data_numpy[:, :, 20]
            # let spine of each frame be the joint coordinate center
            data_numpy = data_numpy - data_numpy[:, :, 20:21]
            # ## works well with bone, but has negative effect with joint and distance gate
            data_numpy[:, :, 20] = trajectory

        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        if self.random_miss:    
            # it is only for robustness test, set false for normal training/testing. Only set true for robustness test at test phrase after training
            miss_num = 0
            channels, origin_t, num_joint, num_people = np.shape(data_numpy) #input shape: C, T, V, M
            t_index = np.arange(origin_t)
            filter = np.random.choice(t_index,size=miss_num, replace=False) 
            indices = np.argwhere(np.isin(t_index,filter))
            valid_t_index = np.delete(t_index,indices)
            data_numpy = data_numpy[:,valid_t_index,:,:].transpose(3,1,2,0)  # M,valid_T,V,C
            data_rescaled = np.zeros((num_people, origin_t, num_joint, channels))
            for i in range(num_people):
                data_rescaled[i] = cv2.resize(data_numpy[i], (num_joint, origin_t), interpolation=cv2.INTER_LINEAR) # use linear interpolation to fill the missing data
            data_numpy = data_rescaled.transpose(3,1,2,0) # back to C,T,V,M      

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


