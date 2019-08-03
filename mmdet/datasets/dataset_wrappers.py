import numpy as np
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .registry import DATASETS


# @DATASETS.register_module
# class xxConcatDataset(_ConcatDataset):
#     """A wrapper of concatenated dataset.
#
#     Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
#     concat the group flag for image aspect ratio.
#
#     Args:
#         datasets (list[:obj:`Dataset`]): A list of datasets.
#     """
#
#     def __init__(self, datasets):
#         super(ConcatDataset, self).__init__(datasets)
#         self.CLASSES = datasets[0].CLASSES
#         if hasattr(datasets[0], 'flag'):
#             flags = []
#             for i in range(0, len(datasets)):
#                 flags.append(datasets[i].flag)
#             self.flag = np.concatenate(flags)

@DATASETS.register_module
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        if hasattr(datasets[0], 'flag'):
            flags = []
            for i in range(0, len(datasets)):
                flags.append(datasets[i].flag)
            self.flag = np.concatenate(flags)

        if hasattr(datasets[0], 'class_indices'):
            lst = [[] for i in range(len(self.CLASSES))]
            ind_offset = 0
            for i in range(0, len(datasets)):
                for j in range(0, len(self.CLASSES)):
                    lst[j].append(datasets[i].class_indices[j] + ind_offset)
                ind_offset += len(datasets[i])
            cls_to_ind = [[] for i in range(len(self.CLASSES))]
            for i in range(0, len(self.CLASSES)):
                cls_to_ind[i] = np.concatenate(lst[i])
            self.cls_to_ind = cls_to_ind

@DATASETS.register_module
class RepeatDataset(object):
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        if hasattr(self.dataset, 'flag'):
            self.flag = np.tile(self.dataset.flag, times)

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        return self.times * self._ori_len
