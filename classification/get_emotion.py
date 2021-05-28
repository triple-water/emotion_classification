import os, time
import torch
import numpy as np
import torch.nn.functional as F
import classification.my_models as my_models
from torch.utils.data import DataLoader

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


class RawEEGDataset(torch.utils.data.Dataset):
    def __init__(self, feature_list):
        self.feature_list = feature_list

    def __getitem__(self, index):
        test = torch.from_numpy(self.feature_list[index])
        feature = F.normalize(torch.from_numpy(self.feature_list[index]).float(), p=2, dim=1)
        # feature = feature[:,np.newaxis,:] # 改变这里可以改输入维度中1 的位置
        feature = feature[np.newaxis, :]
        return feature

    def __len__(self):
        return len(self.feature_list)


def get_emotion_value(data):
    base_path = os.path.dirname(os.path.realpath(__file__))
    data = RawEEGDataset(data)
    test_data_loader = DataLoader(data, batch_size=1, shuffle=False,
                                  drop_last=True)
    model = my_models.inception_se_Final(num_classes=2, batch_size=1, inputsize=62 * 200, hiden=64, dropout_rate=0.2,
                                         hiden2=32)
    model_dict_path = base_path + os.path.sep + "models" + os.path.sep + "BEST_checkpoint.tar"
    model_dict = torch.load(model_dict_path,map_location='cpu')
    model.load_state_dict(model_dict, False)
    model.to(device)
    with torch.no_grad():
        emotion_list = []
        for x_test in test_data_loader:
            x_test = x_test.to(device)
            model.eval()
            yhat_pair, yhat = model(x_test)
            pred = yhat.max(1)[1]
            pred = pred.cpu().numpy()
            emotion_list.append(pred)
        emotion = np.hstack(emotion_list)
        return emotion


if __name__ == '__main__':
    t = time.time()
    # a = np.random.random((50, 14, 256))
    a = np.zeros([50, 14, 256])
    emotion = get_emotion_value(a)
    print(emotion)
    print(time.time() - t)
