import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
def get_emotion(data):
    data = RawEEGDataset(data)
    test_data_loader = DataLoader(data, batch_size=1, shuffle=False,
                                  drop_last=True)
    model = torch.load(r'H:\MAX_MODEL\max_model_2.pt')
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
    data = np.zeros([100, 14, 256])
    emotion = get_emotion(data)
    print(emotion)