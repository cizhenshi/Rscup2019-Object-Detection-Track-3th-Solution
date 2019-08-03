from __future__ import division

import math
import torch
import numpy as np
from icecream import ic
from mmcv.runner.utils import get_dist_info
from torch.utils.data import Sampler
from torch.utils.data import DistributedSampler as _DistributedSampler


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


# class GroupSampler(Sampler):
#
#     def __init__(self, dataset, samples_per_gpu=1):
#         assert hasattr(dataset, 'flag')
#         self.dataset = dataset
#         self.samples_per_gpu = samples_per_gpu
#         self.flag = dataset.flag.astype(np.int64)
#         self.group_sizes = np.bincount(self.flag)
#         self.num_samples = 0
#         for i, size in enumerate(self.group_sizes):
#             self.num_samples += int(np.ceil(
#                 size / self.samples_per_gpu)) * self.samples_per_gpu
#
#     def __iter__(self):
#         indices = []
#         for i, size in enumerate(self.group_sizes):
#             if size == 0:
#                 continue
#             indice = np.where(self.flag == i)[0]
#             assert len(indice) == size
#             np.random.shuffle(indice)
#             num_extra = int(np.ceil(size / self.samples_per_gpu)
#                             ) * self.samples_per_gpu - len(indice)
#             indice = np.concatenate([indice, indice[:num_extra]])
#             indices.append(indice)
#         indices = np.concatenate(indices)
#         indices = [
#             indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
#             for i in np.random.permutation(
#                 range(len(indices) // self.samples_per_gpu))
#         ]
#         indices = np.concatenate(indices)
#         indices = indices.astype(np.int64).tolist()
#         assert len(indices) == self.num_samples
#         return iter(indices)
#
#     def __len__(self):
#         return self.num_samples


class GroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1, samples_per_cls=1000,):
        assert hasattr(dataset, 'cls_to_ind')
        assert hasattr(dataset, 'flag')
        self.dataset = dataset

        self.samples_per_cls = samples_per_cls
        self.cls_to_ind = dataset.cls_to_ind
        for i in range(len(self.cls_to_ind)):
            self.cls_to_ind[i] = self.cls_to_ind[i].astype(np.int64)
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.weight = np.ones(18)
        self.weight[16] = 0.25
        self.weight[10] = 0.6
        self.weight[9] = 0.3
        self.weight = np.array(self.weight, np.int32)

    def __iter__(self):
        indices = np.array([])
        for i in range(len(self.cls_to_ind)):
            indices = np.concatenate([indices, np.random.choice(self.cls_to_ind[i], self.weight[
                i]*self.samples_per_cls)])
        np.random.shuffle(indices)
        ind1 = indices.copy()
        print(ind1.shape)
        ind1 = np.array(ind1, np.int32)
        indices = []
        flag = self.flag[ind1]
        group_sizes = np.bincount(flag)
        num_samples = 0
        for i, size in enumerate(group_sizes):
            num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu

        for i, size in enumerate(group_sizes):
            if size == 0:
                continue
            indice = np.where(flag == i)[0]
            assert len(indice) == size
            np.random.shuffle(indice)
            num_extra = int(np.ceil(size / self.samples_per_gpu)
                            ) * self.samples_per_gpu - len(indice)
            indice = np.concatenate([indice, indice[:num_extra]])
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = torch.from_numpy(indices).long()
        assert len(indices) == num_samples
        return iter(ind1[indices])

    def __len__(self):
        return self.num_samples


class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 samples_per_cls=1000):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        assert hasattr(self.dataset, 'flag')
        assert hasattr(dataset, 'cls_to_ind')
        self.samples_per_cls = samples_per_cls
        self.cls_to_ind = dataset.cls_to_ind
        for i in range(len(self.cls_to_ind)):
            self.cls_to_ind[i] = self.cls_to_ind[i].astype(np.int64)
        self.flag = self.dataset.flag.astype(np.int64)
        self.weight = np.ones(18)
        self.weight[16] = 0.25
        self.weight[10] = 0.6
        self.weight[9] = 0.3
        self.weight = np.array(self.weight, np.int32)
        self.len = int(np.sum(self.weight*samples_per_cls)/self.num_replicas)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        np.random.seed(self.epoch)

        indices = np.array([])
        for i in range(len(self.cls_to_ind)):
            indices = np.concatenate([indices, np.random.choice(self.cls_to_ind[i], self.weight[i] *
                                                                self.samples_per_cls)])
        np.random.shuffle(indices)
        ind1 = indices.copy()
        ind1 = np.array(ind1, np.int32)
        flag = self.flag[ind1]
        group_sizes = np.bincount(flag)
        num_samples = 0
        indices = []
        ic(group_sizes)
        for i, j in enumerate(group_sizes):
            num_samples += int(
                math.ceil(group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        total_size = num_samples * self.num_replicas

        for i, size in enumerate(group_sizes):
            if size > 0:
                indice = np.where(flag == i)[0]
                assert len(indice) == size
                indice = indice[list(torch.randperm(int(size),
                                                    generator=g))].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                indice += indice[:extra]
                indices += indice

        assert len(indices) == total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = num_samples * self.rank
        ic(self.rank)
        ic(offset)
        ic(offset+num_samples)
        indices = indices[offset:offset + num_samples]
        assert len(indices) == num_samples

        return iter(ind1[indices])

    def __len__(self):
        return self.len

    def set_epoch(self, epoch):
        self.epoch = epoch



# class DistributedGroupSampler(Sampler):
#     """Sampler that restricts data loading to a subset of the dataset.
#     It is especially useful in conjunction with
#     :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
#     process can pass a DistributedSampler instance as a DataLoader sampler,
#     and load a subset of the original dataset that is exclusive to it.
#     .. note::
#         Dataset is assumed to be of constant size.
#     Arguments:
#         dataset: Dataset used for sampling.
#         num_replicas (optional): Number of processes participating in
#             distributed training.
#         rank (optional): Rank of the current process within num_replicas.
#     """
#
#     def __init__(self,
#                  dataset,
#                  samples_per_gpu=1,
#                  num_replicas=None,
#                  rank=None):
#         _rank, _num_replicas = get_dist_info()
#         if num_replicas is None:
#             num_replicas = _num_replicas
#         if rank is None:
#             rank = _rank
#         self.dataset = dataset
#         self.samples_per_gpu = samples_per_gpu
#         self.num_replicas = num_replicas
#         self.rank = rank
#         self.epoch = 0
#
#         assert hasattr(self.dataset, 'flag')
#         self.flag = self.dataset.flag
#         self.group_sizes = np.bincount(self.flag)
#
#         self.num_samples = 0
#         for i, j in enumerate(self.group_sizes):
#             self.num_samples += int(
#                 math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
#                           self.num_replicas)) * self.samples_per_gpu
#         self.total_size = self.num_samples * self.num_replicas
#
#     def __iter__(self):
#         # deterministically shuffle based on epoch
#         g = torch.Generator()
#         g.manual_seed(self.epoch)
#
#         indices = []
#         for i, size in enumerate(self.group_sizes):
#             if size > 0:
#                 indice = np.where(self.flag == i)[0]
#                 assert len(indice) == size
#                 indice = indice[list(torch.randperm(int(size),
#                                                     generator=g))].tolist()
#                 extra = int(
#                     math.ceil(
#                         size * 1.0 / self.samples_per_gpu / self.num_replicas)
#                 ) * self.samples_per_gpu * self.num_replicas - len(indice)
#                 indice += indice[:extra]
#                 indices += indice
#
#         assert len(indices) == self.total_size
#
#         indices = [
#             indices[j] for i in list(
#                 torch.randperm(
#                     len(indices) // self.samples_per_gpu, generator=g))
#             for j in range(i * self.samples_per_gpu, (i + 1) *
#                            self.samples_per_gpu)
#         ]
#
#         # subsample
#         offset = self.num_samples * self.rank
#         indices = indices[offset:offset + self.num_samples]
#         assert len(indices) == self.num_samples
#
#         return iter(indices)
#
#     def __len__(self):
#         return self.num_samples
#
#     def set_epoch(self, epoch):
#         self.epoch = epoch
