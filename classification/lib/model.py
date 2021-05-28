import torch
import torch.nn as nn
import torch.nn.functional as F
class Bottleneck(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, in_planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = F.relu(out)
        return out
class CNN_LSTM_V2(nn.Module):
    def __init__(self,num_classes,batch_size,inputsize,hiden,dropout_rate,hiden2):
        super(CNN_LSTM_V2, self).__init__()
        self.inception_window = [0.3, 0.25, 0.125, 0.0625, 0.03125]
        self.sampling_rate = 200
        self.num_T = 5
        self.batch_size = batch_size
        self.dropout = dropout_rate
        self.Tception1 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[0] * self.sampling_rate)), stride=1, padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[1] * self.sampling_rate)), stride=1, padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[2] * self.sampling_rate)), stride=1, padding=0),

            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception4 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[3] * self.sampling_rate)), stride=1,
                      padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception5 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[4] * self.sampling_rate)), stride=1,
                      padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
        self.conv1 = nn.Conv2d(1,9,kernel_size=(1,1),padding=0)
        self.conv2 = nn.Conv2d(1,9, kernel_size=(1,128),padding=1)
        self.conv3 = nn.Conv2d(1,1,kernel_size=(1,1),padding=0)
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        self.fc1 = nn.Linear(19840,1024)
        self.fc2 = nn.Linear(1024,64)
        self.fc3 = nn.Linear(64,3)
    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)
class EEGNet(nn.Module):
    def __init__(self, n_classes=3, channels=14, samples=200,
                 dropoutRate=0.5, kernelLength=100, kernelLength2=16, F1=8,
                 D=2, F2=16):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.samples = samples
        self.n_classes = n_classes
        self.channels = channels
        self.kernelLength = kernelLength
        self.kernelLength2 = kernelLength2
        self.dropoutRate = dropoutRate

        self.blocks = self.InitialBlocks(dropoutRate)
        self.blockOutputSize = self.CalculateOutSize(self.blocks, channels, samples)
        self.classifierBlock = self.ClassifierBlock(self.F2 * self.blockOutputSize[1], n_classes)

    def InitialBlocks(self, dropoutRate, *args, **kwargs):
        block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernelLength), stride=1, padding=(0, self.kernelLength // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),

            # DepthwiseConv2D =======================
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.channels, 1), max_norm=1, stride=1, padding=(0, 0),
                                 groups=self.F1, bias=False),
            # ========================================

            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(p=dropoutRate))
        block2 = nn.Sequential(
            # SeparableConv2D =======================
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernelLength2), stride=1,
                      padding=(0, self.kernelLength2 // 2), bias=False, groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), padding=(0, 0), groups=1, bias=False, stride=1),
            # ========================================

            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=(1, 8)),
            nn.Dropout(p=dropoutRate))
        return nn.Sequential(block1, block2)


    def ClassifierBlock(self, inputSize, n_classes):
        return nn.Sequential(
            nn.Linear(inputSize, n_classes, bias=False),
            nn.Softmax(dim=1))

    def CalculateOutSize(self, model, channels, samples):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, 1, channels, samples)
        model.eval()
        out = model(data).shape
        return out[2:]

        x = x.view(self.batch_size,31*self.num_T,-1) # x : 128 x 5 x 62 x 146 batch_size x num_t x channels x data
        # x = torch.squeeze(out)
        x,(hn,cn) = self.lstm1(x) # 128 * 155 * 292
        # x,(hn,cn) = self.lstm2(x)
        x1 = x.contiguous().view(x.size()[0], -1) # 128 * 155 * 128  -- > 128 * 19840


        x = F.relu(self.fc1(x1))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout)
        x = self.fc3(x)
        x = F.softmax(x,dim=-1)
        return x1,x
    def forward(self, x):
        x = self.blocks(x)
        x = x.view(x.size()[0], -1)  # Flatten
        x = self.classifierBlock(x)

        return x,x
class icse_emotiv(nn.Module):
    def __init__(self,num_classes,batch_size,inputsize,hiden,dropout_rate,hiden2):
        super(icse_emotiv, self).__init__()
        self.inception_window = [0.3, 0.25, 0.125, 0.0625, 0.03125]
        # self.inception_window = [0.25, 0.125, 0.0625]
        self.sampling_rate = 200
        self.num_T = 9
        self.channel = 14
        self.se = 4
        self.batch_size = batch_size
        self.dropout = dropout_rate
        self.se_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.num_T, self.num_T//self.se, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.num_T // self.se, self.num_T, kernel_size=1),
            nn.Sigmoid()
        )
        # nn.BatchNorm2d(self.num_T//self.se),
        # nn.BatchNorm2d(self.num_T),
        self.Tception1 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[0] * self.sampling_rate)), stride=1, padding=0),  # 2*3*3
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[1] * self.sampling_rate)), stride=1, padding=0),  # 2*3*3*3*3*3
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[2] * self.sampling_rate)), stride=1, padding=0),  # 3*3*3... *3 （12个）

            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception4 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[3] * self.sampling_rate)), stride=1,
                      padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception5 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[4] * self.sampling_rate)), stride=1,
                      padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
        self.conv1 = nn.Conv2d(1,9,kernel_size=(1,1),padding=0)
        self.conv2 = nn.Conv2d(1,9, kernel_size=(1,128),padding=1)
        self.conv3 = nn.Conv2d(1,1,kernel_size=(1,1),padding=0)

        self.BN_t = nn.BatchNorm2d(self.num_T)

        self.lstm1 = nn.LSTM(input_size=83,num_layers=2, hidden_size=hiden,dropout=dropout_rate,bidirectional=True,batch_first=True)
        self.bottle_neck = Bottleneck(1, self.num_T)
        # self.lstm2 = nn.LSTM(input_size=hiden,num_layers=2,hidden_size=hiden2,dropout=dropout_rate,bidirectional=False)
        self.fc1 = nn.Linear(16128,2048)
        self.fc2 = nn.Linear(2048,128)
        self.fc3 = nn.Linear(128,num_classes)
    def forward(self,x):


        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x,2)

        # x = self.conv(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x,2)
        y = self.Tception1(x)  # x: torch.Size([128,1,62,205])
        y_se = self.se_net(y)
        out = y * y_se.expand_as(y)

        y = self.Tception2(x)
        y_se = self.se_net(y)
        y = y * y_se.expand_as(y)
        out = torch.cat((out, y), dim=-1)

        y = self.Tception3(x)
        y_se = self.se_net(y)
        y = y * y_se.expand_as(y)
        out = torch.cat((out, y), dim=-1)

        y = self.Tception4(x)
        y_se = self.se_net(y)
        y = y * y_se.expand_as(y)
        out = torch.cat((out, y), dim=-1)

        y = self.Tception5(x)
        y_se = self.se_net(y)
        y = y * y_se.expand_as(y)
        out = torch.cat((out, y), dim=-1)

        x = self.BN_t(out)



        x = x.view(self.batch_size,126,-1) # x : 24 x 1 x 31 x 100  batch_size x num_t x channels x data
        # x = torch.squeeze(out)
        x,(hn,cn) = self.lstm1(x)
        # x,(hn,cn) = self.lstm2(x)
        x = x.contiguous().view(x.size()[0], -1)


        x2 = F.relu(self.fc1(x))
        x2 = F.dropout(x2, self.dropout)
        x2 = F.relu(self.fc2(x2))
        x2 = F.dropout(x2, self.dropout)
        x2 = self.fc3(x2)
        x2 = F.softmax(x2, dim=-1)
        return x, x2
class CNN_LSTM_DNM(nn.Module):
    def __init__(self,num_classes,batch_size,inputsize,hiden,dropout_rate,hiden2):
        super(CNN_LSTM_DNM, self).__init__()
        self.inception_window = [0.3, 0.25, 0.125, 0.0625, 0.03125]
        self.sampling_rate = 200
        self.num_T = 5
        self.batch_size = batch_size
        self.dropout = dropout_rate
        self.Tception1 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[0] * self.sampling_rate)), stride=1, padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[1] * self.sampling_rate)), stride=1, padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[2] * self.sampling_rate)), stride=1, padding=0),

            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception4 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[3] * self.sampling_rate)), stride=1,
                      padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception5 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[4] * self.sampling_rate)), stride=1,
                      padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
        self.conv1 = nn.Conv2d(1,9,kernel_size=(1,1),padding=0)
        self.conv2 = nn.Conv2d(1,9, kernel_size=(1,128),padding=1)
        self.conv3 = nn.Conv2d(1,1,kernel_size=(1,1),padding=0)

        self.BN_t = nn.BatchNorm2d(self.num_T)
        self.BN_t2 = nn.BatchNorm1d(9)

        self.lstm1 = nn.LSTM(input_size=284,num_layers=2, hidden_size=hiden,dropout=dropout_rate,bidirectional=False,batch_first=True)

        # self.lstm2 = nn.LSTM(input_size=hiden,num_layers=2,hidden_size=hiden2,dropout=dropout_rate,bidirectional=False)
        self.fc1 = nn.Linear(9920,512)
        self.fc2 = nn.Linear(512,9)
        self.fc3 = nn.Linear(64,num_classes)
        self.DNM = Dnm_Net_v2(M=9,max_Neuron=9,BN_internal=3,k1=9,k2=10)
    def forward(self,x):

        #
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x,2)
        #
        # x = self.conv3(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x,2)
        y = self.Tception1(x)  # x: torch.Size([24,1,31,100])
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception4(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception5(x)
        out = torch.cat((out, y), dim=-1)

        # y = self.conv1(x)
        # y = F.relu(y)
        # y = F.avg_pool2d(y, kernel_size=(1, 1))
        # out = torch.cat((out, y), dim=-1)
        x = self.BN_t(out)


        x = x.view(self.batch_size,31*self.num_T,-1) # x : 24 x 1 x 31 x 100  batch_size x num_t x channels x data
        # x = torch.squeeze(out)
        x,(hn,cn) = self.lstm1(x)

        # x,(hn,cn) = self.lstm2(x)
        x = x.contiguous().view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout)
        x = self.BN_t2(x)
        x = self.DNM(x)
        x = F.softmax(x,dim=-1)
        return x
class Dnm_Net(torch.nn.Module): ## 这是第一个版本的，存在梯度消失的问题
    def __init__(self, M, row,k1,k2):
        """
        M ： 待训练的数据行数
        row ： 待训练的数据列数
        k1 ：ks is a positive constant whose value is usually an integer between 1
            and 10.
        k2 ：kc  is a positive constant whose value is usually
             an integer between 1 and 20.
        """
        super(Dnm_Net, self).__init__()
        self.W_orig = nn.Parameter(torch.rand(M, row - 1,  dtype=torch.float32))
        self.q_orig = nn.Parameter(torch.rand(M, row - 1,  dtype=torch.float32))
        self.W2_orig = nn.Parameter(torch.rand(M, row - 1, dtype=torch.float32))
        self.q2_orig = nn.Parameter(torch.rand(M, row - 1,  dtype=torch.float32))
        self.W3_orig = nn.Parameter(torch.rand(M, row - 1, dtype=torch.float32))
        self.q3_orig = nn.Parameter(torch.rand(M, row - 1, dtype=torch.float32))
        self.soma_layer_parameter = nn.Parameter(torch.randn(1))
        self.M = M
        self.k1 = k1
        self.k2 = k2
    def forward(self, x):
        """
        此方法中实现了DNM的前向传播，输入X为二维矩阵
        """
        # standardScaler = MinMaxScaler()
        # standardScaler.fit(x.cpu().detach().numpy())
        # x = standardScaler.transform(x.cpu().detach().numpy())
        X_orig = torch.as_tensor(x,device='cuda',dtype=torch.float32)
        X_orig = X_orig.unsqueeze(dim=1)
        X_orig = X_orig.expand(X_orig.shape[0], self.M, X_orig.shape[2])  # 若输入X为多维矩阵，此处应修改
        y_temp_orig = F.leaky_relu(self.k1 * 1.0 * (torch.mul(X_orig, self.W_orig) - self.q_orig))
        y_temp_orig2 = F.leaky_relu(self.k1 * 1.0 * (torch.mul(X_orig, self.W2_orig) - self.q2_orig))
        y_temp_orig3 = F.leaky_relu(self.k1 * 1.0 * (torch.mul(X_orig, self.W3_orig) - self.q3_orig))
        y_orig_prod = torch.prod(y_temp_orig, dim=2, keepdim=False)  # 将不同特征的维度相乘 此处因为sigmoid激活导致数据分布在0-1之间，在特征多时会导致相乘后为0  方法一：前面降维：lstm降维或全连接层降维，方法二：自剪特征降维
        y_orig2_prod = torch.prod(y_temp_orig2, dim=2, keepdim=False)
        y_orig3_prod = torch.prod(y_temp_orig3, dim=2, keepdim=False)
        y1_orig_sum = torch.sum(y_orig_prod, dim=1, keepdim=True)  # 并相加
        y1_orig2_sum = torch.sum(y_orig2_prod, dim=1, keepdim=True)  # 并相加
        y1_orig3_sum = torch.sum(y_orig3_prod, dim=1, keepdim=True)  # 并相加
        y2_orig_final = torch.sigmoid(self.k2 * 1.0 * (y1_orig_sum - self.soma_layer_parameter))
        y2_orig2_final = torch.sigmoid(self.k2 * 1.0 * (y1_orig2_sum - self.soma_layer_parameter))
        y2_orig3_final = torch.sigmoid(self.k2 * 1.0 * (y1_orig3_sum - self.soma_layer_parameter))
        y_hat  = torch.cat([y2_orig_final,y2_orig2_final,y2_orig3_final],dim=1)
        print("W的梯度：{}    q的梯度: {}".format(self.W_orig.grad,self.q_orig.grad))
        return y_hat
class Dnm_Net_v2(torch.nn.Module):
    def __init__(self, M,max_Neuron,BN_internal,k1,k2):
        """
        M ： 待训练的数据batch_size
        row : 数据的特征数
        max_Neuron ： 最大神经元容量即每个神经元包含的特征数
        BN_internal : 经过多少特征后进行一次batch norm
        k1 ：ks is a positive constant whose value is usually an integer between 1
            and 10.
        k2 ：kc  is a positive constant whose value is usually
             an integer between 1 and 20.
        """
        super(Dnm_Net_v2, self).__init__()
        self.W_orig = nn.Parameter(torch.rand(M, max_Neuron,  dtype=torch.float32))
        self.q_orig = nn.Parameter(torch.rand(M, max_Neuron,  dtype=torch.float32))

        self.W2_orig = nn.Parameter(torch.rand(M, max_Neuron, dtype=torch.float32))
        self.q2_orig = nn.Parameter(torch.rand(M, max_Neuron, dtype=torch.float32))
        self.W3_orig = nn.Parameter(torch.rand(M, max_Neuron, dtype=torch.float32))
        self.q3_orig = nn.Parameter(torch.rand(M, max_Neuron, dtype=torch.float32))

        self.max_Neuron = max_Neuron
        self.BN_num = max_Neuron/BN_internal
        self.BN_internal = BN_internal
        self.M = M
        self.k1 = k1
        self.k2 = k2
    def forward(self, x):
        """
        此方法中实现了DNM的前向传播，输入X为二维矩阵
        """
        X_orig = torch.as_tensor(x,device='cuda',dtype=torch.float32)
        # x_split = torch.split(X_orig,self.max_Neuron,dim=1)
        y = torch.tensor([],device='cuda')
        X_orig = X_orig.unsqueeze(dim=1)
        X_orig = X_orig.expand(X_orig.shape[0], self.M, X_orig.shape[2])  # 若输入X为多维矩阵，此处应修改
        y_temp_orig = F.leaky_relu(self.k1 * 1.0 * (torch.mul(X_orig, self.W_orig) - self.q_orig))
        y_temp_orig2 = F.leaky_relu(self.k1 * 1.0 * (torch.mul(X_orig, self.W2_orig) - self.q2_orig))
        y_temp_orig3 = F.leaky_relu(self.k1 * 1.0 * (torch.mul(X_orig, self.W3_orig) - self.q3_orig))

        y_split = torch.split(y_temp_orig,self.BN_internal,dim=2)
        batchnorm = nn.BatchNorm1d(9)
        batchnorm.cuda()
        y_group_out = torch.ones(y_temp_orig.shape[0],y_temp_orig.shape[1],1,device='cuda')

        y_split2 = torch.split(y_temp_orig2, self.BN_internal, dim=2)
        batchnorm2 = nn.BatchNorm1d(9)
        y_group_out2 = torch.ones(y_temp_orig2.shape[0], y_temp_orig2.shape[1], 1)
        y_split3 = torch.split(y_temp_orig3, self.BN_internal, dim=2)
        batchnorm3 = nn.BatchNorm1d(9)
        y_group_out3 = torch.ones(y_temp_orig3.shape[0], y_temp_orig3.shape[1], 1)

        for i in range(len(y_split)):
            y_split_temp = y_split[i]
            y_group_prod = y_group_out * torch.prod(y_split_temp, dim=2, keepdim=True) # 将不同特征的维度相乘 此处因为sigmoid激活导致数据分布在0-1之间，在特征多时会导致相乘后为0  方法一：前面降维：lstm降维或全连接层降维，方法二：自剪特征降维
            y_group_out = batchnorm(y_group_prod)
        y_orig_prod = torch.squeeze(y_group_out,dim=2)
        y1_orig_sum = torch.sum(y_orig_prod, dim=1, keepdim=True)  # 并相加
        y2_orig_final = torch.sigmoid(self.k2 * 1.0 * (y1_orig_sum - 0.5))
        y = torch.cat([y,y2_orig_final],dim=1)

        for i in range(len(y_split2)):
            y_split_temp2 = y_split2[i]
            y_group_prod2 = y_group_out * torch.prod(y_split_temp2, dim=2, keepdim=True) # 将不同特征的维度相乘 此处因为sigmoid激活导致数据分布在0-1之间，在特征多时会导致相乘后为0  方法一：前面降维：lstm降维或全连接层降维，方法二：自剪特征降维
            y_group_out2 = batchnorm(y_group_prod2)
        y_orig_prod2 = torch.squeeze(y_group_out2,dim=2)
        y1_orig_sum2 = torch.sum(y_orig_prod2, dim=1, keepdim=True)  # 并相加
        y2_orig_final2 = torch.sigmoid(self.k2 * 1.0 * (y1_orig_sum2 - 0.5))
        y2 = torch.cat([y,y2_orig_final2],dim=1)

        for i in range(len(y_split3)):
            y_split_temp3 = y_split3[i]
            y_group_prod3 = y_group_out * torch.prod(y_split_temp3, dim=2, keepdim=True) # 将不同特征的维度相乘 此处因为sigmoid激活导致数据分布在0-1之间，在特征多时会导致相乘后为0  方法一：前面降维：lstm降维或全连接层降维，方法二：自剪特征降维
            y_group_out3 = batchnorm(y_group_prod3)
        y_orig_prod3 = torch.squeeze(y_group_out3,dim=2)
        y1_orig_sum3 = torch.sum(y_orig_prod3, dim=1, keepdim=True)  # 并相加
        y2_orig_final3 = torch.sigmoid(self.k2 * 1.0 * (y1_orig_sum3 - 0.5))
        y3 = torch.cat([y,y2_orig_final3],dim=1)

        y_hat  = torch.cat([y2_orig_final,y2_orig_final2,y2_orig_final3],dim=1)
        # print("W的梯度：{}    q的梯度: {}".format(self.W_orig.grad,self.q_orig.grad))
        return y_hat
class CNN_LSTM_FPZ(nn.Module):
    def __init__(self,num_classes,batch_size,inputsize,hiden,dropout_rate,hiden2):
        super(CNN_LSTM_FPZ, self).__init__()
        self.inception_window = [0.3, 0.25, 0.125, 0.0625, 0.03125]
        self.sampling_rate = 200
        self.num_T = 5
        self.batch_size = batch_size
        self.dropout = dropout_rate
        self.Tception1 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[0] * self.sampling_rate)), stride=1, padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[1] * self.sampling_rate)), stride=1, padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[2] * self.sampling_rate)), stride=1, padding=0),

            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception4 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[3] * self.sampling_rate)), stride=1,
                      padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception5 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[4] * self.sampling_rate)), stride=1,
                      padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
        self.conv1 = nn.Conv2d(1,9,kernel_size=(1,1),padding=0)
        self.conv2 = nn.Conv2d(1,9, kernel_size=(1,128),padding=1)
        self.conv3 = nn.Conv2d(1,1,kernel_size=(1,1),padding=0)

        self.BN_t = nn.BatchNorm2d(self.num_T)

        self.lstm1 = nn.LSTM(input_size=4402,num_layers=2, hidden_size=hiden,dropout=dropout_rate,bidirectional=True,batch_first=True)

        # self.lstm2 = nn.LSTM(input_size=hiden,num_layers=2,hidden_size=hiden2,dropout=dropout_rate,bidirectional=False)
        self.fc1 = nn.Linear(1280,64)
        # self.fc2 = nn.Linear(2048,64)
        self.fc3 = nn.Linear(64,num_classes)
    def forward(self,x):

        #
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x,2)
        #
        # x = self.conv3(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x,2)
        y = self.Tception1(x)  # x: torch.Size([24,1,31,100])
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception4(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception5(x)
        out = torch.cat((out, y), dim=-1)

        # y = self.conv1(x)
        # y = F.relu(y)
        # y = F.avg_pool2d(y, kernel_size=(1, 1))
        # out = torch.cat((out, y), dim=-1)
        x = self.BN_t(out)


        # x = x.view(self.batch_size,31*self.num_T,-1) # x : 24 x 1 x 31 x 100  batch_size x num_t x channels x data
        x = x.view(self.batch_size, 10, -1)
        # x = torch.squeeze(out)
        x,(hn,cn) = self.lstm1(x)
        # x,(hn,cn) = self.lstm2(x)
        x = x.contiguous().view(x.size()[0], -1)


        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout)
        # x = F.relu(self.fc2(x))
        # x = F.dropout(x, self.dropout)
        x = self.fc3(x)
        x = F.softmax(x,dim=-1)
        return x
class CNN_LSTM_DE(nn.Module):
    def __init__(self,num_classes,batch_size,inputsize,hiden,dropout_rate,hiden2):
        super(CNN_LSTM_DE, self).__init__()
        self.inception_window = [0.5, 0.4, 0.3,0.25]
        self.sampling_rate = 5
        self.num_T = 3
        self.batch_size = batch_size
        self.dropout = dropout_rate
        self.Tception1 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[0] * self.sampling_rate)), stride=1, padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[1] * self.sampling_rate)), stride=1, padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[2] * self.sampling_rate)), stride=1, padding=0),

            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception4 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[3] * self.sampling_rate)), stride=1,
                      padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        # self.Tception5 = nn.Sequential(
        #     nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[4] * self.sampling_rate)), stride=1,
        #               padding=0),
        #     nn.BatchNorm2d(self.num_T),
        #     nn.ReLU(),
        #     nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
        self.conv1 = nn.Conv2d(1,9,kernel_size=(1,1),padding=0)
        self.conv2 = nn.Conv2d(1,9, kernel_size=(1,128),padding=1)
        self.conv3 = nn.Conv2d(1,1,kernel_size=(1,1),padding=0)

        self.BN_t = nn.BatchNorm2d(self.num_T)

        self.lstm1 = nn.LSTM(input_size=3,num_layers=3, hidden_size=hiden,dropout=dropout_rate,bidirectional=True,batch_first=True)

        # self.lstm2 = nn.LSTM(input_size=hiden,num_layers=2,hidden_size=hiden2,dropout=dropout_rate,bidirectional=False)
        self.fc1 = nn.Linear(7936,128)
        # self.fc2 = nn.Linear(1024,64)
        self.fc3 = nn.Linear(128,num_classes)
    def forward(self,x):


        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x,2)

        # x = self.conv3(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x,2)
        y = self.Tception1(x)  # x: torch.Size([64,1,62,5])
        out = y
        # y = self.Tception2(x)
        # out = torch.cat((out, y), dim=-1)
        # y = self.Tception3(x)
        # out = torch.cat((out, y), dim=-1)
        # y = self.Tception4(x)
        # out = torch.cat((out, y), dim=-1)
        # # y = self.Tception5(x)
        # out = torch.cat((out, y), dim=-1)

        # y = self.conv1(x)
        # y = F.relu(y)
        # y = F.avg_pool2d(y, kernel_size=(1, 1))
        # out = torch.cat((out, y), dim=-1)
        x = self.BN_t(out)


        x = x.view(self.batch_size,62,-1) # x : 24 x 1 x 31 x 100  batch_size x num_t x channels x data
        # x = torch.squeeze(out)
        x,(hn,cn) = self.lstm1(x)
        # x,(hn,cn) = self.lstm2(x)
        x = x.contiguous().view(x.size()[0], -1)


        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout)
        # x = F.relu(self.fc2(x))
        # x = F.dropout(x, self.dropout)
        x = self.fc3(x)
        x = F.softmax(x,dim=-1)
        return x
class Tsception_LSTM(nn.Module):
    def __init__(self,num_classes,batch_size,inputsize,hiden,dropout_rate,hiden2):
        super(Tsception_LSTM, self).__init__()
        self.inception_window = [0.3, 0.25, 0.125, 0.0625, 0.03125]
        self.sampling_rate = 200
        self.num_channels = 62
        self.num_T = 5
        self.num_S = 4
        self.batch_size = batch_size
        self.dropout = dropout_rate
        self.Tception1 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[0] * self.sampling_rate)), stride=1, padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[1] * self.sampling_rate)), stride=1, padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[2] * self.sampling_rate)), stride=1, padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception4 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[3] * self.sampling_rate)), stride=1,
                      padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception5 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[4] * self.sampling_rate)), stride=1,
                      padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
        self.Sception1 = nn.Sequential(
            nn.Conv2d(self.num_T, self.num_S, kernel_size=(int(self.num_channels), 1), stride=1, padding=0),
            nn.BatchNorm2d(self.num_S),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_S), stride=(1, self.num_S)))
        self.Sception2 = nn.Sequential(
            nn.Conv2d(self.num_T, self.num_S, kernel_size=(int(self.num_channels * 0.5), 1), stride=(int(self.num_channels *0.5), 1),
                      padding=0),
            nn.BatchNorm2d(self.num_S),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_S), stride=(1, self.num_S)))
        self.conv1 = nn.Conv2d(1,9,kernel_size=(1,1),padding=0)
        self.conv2 = nn.Conv2d(1,9, kernel_size=(1,128),padding=1)
        self.conv3 = nn.Conv2d(1,1,kernel_size=(1,1),padding=0)

        self.BN_t = nn.BatchNorm2d(self.num_T)
        self.BN_s = nn.BatchNorm2d(self.num_S)
        self.lstm1 = nn.LSTM(input_size=108,num_layers=2, hidden_size=hiden,dropout=dropout_rate,bidirectional=True,batch_first=True)

        # self.lstm2 = nn.LSTM(input_size=hiden,num_layers=2,hidden_size=hiden2,dropout=dropout_rate,bidirectional=False)
        self.fc1 = nn.Linear(19840,1024)
        self.fc2 = nn.Linear(512,64)
        self.fc3 = nn.Linear(64,num_classes)
    def forward(self,x):


        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x,2)

        # x = self.conv(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x,2)
        y = self.Tception1(x)  # x: torch.Size([64,1,62,200])
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception4(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception5(x)
        out = torch.cat((out, y), dim=-1)
        x = self.BN_t(out)

        y = self.Sception1(x)
        out = y
        y = self.Sception2(x)
        x = torch.cat((out, y), dim=2)

        x = x.view(self.batch_size,self.num_S,-1) # x : 24 x 1 x 31 x 100  batch_size x num_t x channels x data
        # x = torch.squeeze(out)
        x,(hn,cn) = self.lstm1(x)
        # x,(hn,cn) = self.lstm2(x)
        x = x.contiguous().view(x.size()[0], -1)


        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, self.dropout)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout)
        x = self.fc3(x)
        x = F.softmax(x,dim=-1)
        return x
class inception_se_Final(nn.Module):
    def __init__(self,num_classes,batch_size,inputsize,hiden,dropout_rate,hiden2):
        super(inception_se_Final, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125, 0.0625, 0.03125]
        # self.inception_window = [0.25, 0.125, 0.0625]
        self.sampling_rate = 256
        self.num_T = 6
        self.channel = 62
        self.se = 2
        self.batch_size = batch_size
        self.dropout = dropout_rate
        self.se_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.num_T, self.num_T//self.se, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.num_T // self.se, self.num_T, kernel_size=1),
            nn.Sigmoid()
        )
        # nn.BatchNorm2d(self.num_T//self.se),
        # nn.BatchNorm2d(self.num_T),
        self.Tception1 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[0] * self.sampling_rate)), stride=1, padding=0),  # 2*3*3
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[1] * self.sampling_rate)), stride=1, padding=0),  # 2*3*3*3*3*3
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[2] * self.sampling_rate)), stride=1, padding=0),  # 3*3*3... *3 （12个）

            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception4 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[3] * self.sampling_rate)), stride=1,
                      padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, self.num_T), stride=(1, self.num_T)))
        self.Tception5 = nn.Sequential(
            nn.Conv2d(1, self.num_T, kernel_size=(1, int(self.inception_window[4] * self.sampling_rate)), stride=1,
                      padding=0),
            nn.BatchNorm2d(self.num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
        self.bottle_neck = Bottleneck(1,self.num_T)
        self.conv1 = nn.Conv2d(1,self.num_T,kernel_size=(1,1),padding=0)
        self.conv2 = nn.Conv2d(self.num_T,self.num_T, kernel_size=(3,3),padding=1)
        self.conv3 = nn.Conv2d(self.num_T,1,kernel_size=(1,1),padding=0)

        self.BN_t = nn.BatchNorm2d(self.num_T)

        self.lstm1 = nn.LSTM(input_size=60,num_layers=2, hidden_size=hiden,dropout=dropout_rate,bidirectional=True,batch_first=True)

        # self.lstm2 = nn.LSTM(input_size=hiden,num_layers=2,hidden_size=hiden2,dropout=dropout_rate,bidirectional=False)
        self.fc1 = nn.Linear(25984,2048)
        self.fc2 = nn.Linear(2048,128)
        self.fc3 = nn.Linear(128,num_classes)
    def forward(self,x):


        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x,2)

        # x = self.conv(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x,2)
        y = self.Tception1(x)  # x: torch.Size([128,62,1,205])
        y_se = self.se_net(y)
        out = y * y_se.expand_as(y)

        y = self.Tception2(x)
        y_se = self.se_net(y)
        y = y * y_se.expand_as(y)
        out = torch.cat((out, y), dim=-1)

        y = self.Tception3(x)
        y_se = self.se_net(y)
        y = y * y_se.expand_as(y)
        out = torch.cat((out, y), dim=-1)

        y = self.Tception4(x)
        y_se = self.se_net(y)
        y = y * y_se.expand_as(y)
        out = torch.cat((out, y), dim=-1)

        y = self.Tception5(x)
        y_se = self.se_net(y)
        y = y * y_se.expand_as(y)
        out = torch.cat((out, y), dim=-1)

        x = self.BN_t(out)



        x = x.view(self.batch_size,203,-1) # x : 24 x 1 x 31 x 100  batch_size x num_t x channels x data
        # x = torch.squeeze(out)
        x,(hn,cn) = self.lstm1(x)
        # x,(hn,cn) = self.lstm2(x)
        x = x.contiguous().view(x.size()[0], -1)


        x2 = F.relu(self.fc1(x))
        x2 = F.dropout(x2, self.dropout)
        x2 = F.relu(self.fc2(x2))
        x2 = F.dropout(x2, self.dropout)
        x2 = self.fc3(x2)
        x2 = F.softmax(x2, dim=-1)
        return x, x2

class TSception(nn.Module):
    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hiden, dropout_rate):
        # input_size: EEG channel x datapoint
        super(TSception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125, 0.0625, 0.03125]
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, int(self.inception_window[0] * sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, int(self.inception_window[1] * sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, int(self.inception_window[2] * sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))

        self.Sception1 = nn.Sequential(
            nn.Conv2d(num_T, num_S, kernel_size=(int(input_size[0]), 1), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))
        self.Sception2 = nn.Sequential(
            nn.Conv2d(num_T, num_S, kernel_size=(int(input_size[0] * 0.5), 1), stride=(int(input_size[0] * 0.5), 1),
                      padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))

        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        size = self.get_size(input_size)

        self.fc1 = nn.Sequential(
            nn.Linear(size[1], hiden),
            nn.ReLU(),
            nn.Dropout(dropout_rate))
        self.fc2 = nn.Sequential(
            nn.Linear(hiden, num_classes),
            nn.LogSoftmax())

    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_final = z
        z = self.Sception2(out)
        out_final = torch.cat((out_final, z), dim=2)
        out = self.BN_s(out_final)
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out,out

    def get_size(self, input_size):
        # here we use and array with the shape being
        # (1(mini-batch),1(convolutional channel),EEG channel,time data point)
        # to simulate the input data and get the output size
        data = torch.ones((1, 1, input_size[0], input_size[1]))

        y = self.Tception1(data)
        out = y
        y = self.Tception2(data)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(data)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_final = z
        z = self.Sception2(out)
        out_final = torch.cat((out_final, z), dim=2)
        out = self.BN_s(out_final)
        out = out.view(out.size()[0], -1)
        return out.size()