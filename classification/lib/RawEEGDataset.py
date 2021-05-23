import torch
import torch.nn.functional as F
import numpy as np
class RawEEGDataset(torch.utils.data.Dataset):
    def __init__(self, feature_list, label_list, desire_shape):
        self.feature_list = feature_list
        self.label_list = label_list
        self.desire_shape = desire_shape

    def __getitem__(self, index):
        # self.feature_list[index] = self.feature_list[index].reshape(self.desire_shape)
        # 1 * 62 * 200，对 200 这个维度进行归一化
        test = torch.from_numpy(self.feature_list[index])
        feature = F.normalize(torch.from_numpy(self.feature_list[index]).float(), p=2, dim=1)
        # feature = feature[:,np.newaxis,:] # 改变这里可以改输入维度中1 的位置
        feature = feature[np.newaxis, :]
        label = torch.tensor(self.label_list[index])
        return feature, label

    def __len__(self):
        return len(self.label_list)