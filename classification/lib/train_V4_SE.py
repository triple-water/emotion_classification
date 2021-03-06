import h5py
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import torch.nn as nn
from pathlib import Path
from RawEEGDataset import *
from optuna.study import TrialState
from torch.utils.data import DataLoader
from model import *
import torch
from sklearn.manifold import TSNE
# from visdom import Visdom
import numpy as np
import torch.nn.functional as F
import optuna
from torch.nn.modules.loss import _Loss
def new_file(test_dir):
    list2 = []
    #列举test_dir目录下的所有文件（名），结果以列表形式返回。
    lists=os.listdir(test_dir)
    for i in lists:
        if i.endswith('.tar'):
            list2.append(i)
    #sort按key的关键字进行升序排序，lambda的入参fn为lists列表的元素，获取文件的最后修改时间，所以最终以文件时间从小到大排序
    #最后对lists元素，按文件修改时间大小从小到大排序。
    list2.sort(key=lambda fn:os.path.getmtime(test_dir+'\\'+fn))
    #获取最新文件的绝对路径，列表中最后一个值,文件夹+文件名
    file_path=os.path.join(test_dir,list2[-1])
    return file_path
name_loss = ['train_loss', 'val_loss']
name_acc = ['train_acc', 'val_acc']

class Pairwise_Loss(_Loss):
    """
    Pairwise loss function.
    Based on: EEG-Based Emotion Recognition with Similarity Learning Network
    """

    def __init__(self,margin=21):
        super(Pairwise_Loss, self).__init__()
        self.margin = margin
    def EuclideanDistances(self,a, b):
        sq_a = a ** 2
        sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
        sq_b = b ** 2
        sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)  # n->[1, n]
        bt = b.t()
        return torch.sqrt(sum_sq_a + sum_sq_b - 2 * a.mm(bt))

    def euclidean_dist(self,x, y):
        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """

        m, n = x.size(0), y.size(0)
        # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        # yy会在最后进行转置的操作
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
        dist.addmm_(beta=1, alpha=-2, mat1=x, mat2=y.t())
        # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist
    def forward(self, output, label):
        loss = []
        a0 = torch.where(label == 0)
        a1 = torch.where(label == 1)
        a2 = torch.where(label == 2)
        output_0, output_1,output_2 = output[a0],output[a1],output[a2]
        euclidean_distance = self.euclidean_dist(output_0, output_1)
        # euclidean_distance = self.EuclideanDistances(output_0,output_1)
        # Is = 1 if label[i] == label[i+1] else 0
        # loss_contrastive = torch.mean(Is * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
        #                               (1-Is) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        loss2 = torch.mean(euclidean_distance)
        return loss2/20

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2, )
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


class TrainModel():
    def __init__(self):
        self.ACC_val = None
        self.accs = None
        self.Loss_val = None
        self.losses = None
        self.data_DE = None
        # self.train_label_DE = None
        # self.val_label_DE = None
        self.test_data_DE = None
        # self.test_label_DE = None
        self.data = None
        self.label = None
        self.train_data = None
        self.train_label = None
        self.val_data = None
        self.val_label = None
        self.test_data = None
        self.test_label = None
        self.train_data_normal = None
        self.loss_lambda = None
        self.val_data_normal = None
        self.test_data_normal = None
        self.result = None
        self.input_shape = None  # should be (eeg_channel, time data point)
        # self.model = 'TSception'
        self.model = 'Tsception_LSTM'
        self.cross_validation = 'Session'  # Subject
        self.sampling_rate = 200
        self.stopepoch = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Parameters: Training process
        self.random_seed = 42
        self.learning_rate = 1e-3
        self.num_epochs = 200
        self.num_class = 3
        self.batch_size = 512
        self.patient = 50  # early stopping patient
        # Parameters: Model
        self.dropout = 0.3
        self.hiden_node = 128
        self.T = 9
        self.S = 6
        self.Lambda = 1e-6

    def load_data(self, path):
        '''
        This is the function to load the data
        Data format : .hdf
        Input : path
                the path of your data
                type = string
        '''
        path = Path(path)
        dataset = h5py.File(path, 'r')
        self.data = np.array(dataset['data'])
        self.label = np.array(dataset['label'])
        self.test_data = np.array(dataset['test_data'])
        self.test_label = np.array(dataset['test_label'])

        print('Data loaded!\n Data shape:[{}], Label shape:[{}], Test_Data shape:[{}], Test_Label shape:[{}]'
              .format(self.data.shape, self.label.shape, self.test_data.shape, self.test_label.shape))

    def get_kfold_data(init, k, i, X, y):

        # 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
        fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）

        val_start = i * fold_size
        if i != k - 1:
            val_end = (i + 1) * fold_size
            X_valid, y_valid = X[val_start:val_end], y[val_start:val_end]
            X_train = np.concatenate((X[0:val_start], X[val_end:]), axis=0)
            y_train = np.concatenate((y[0:val_start], y[val_end:]), axis=0)
        else:  # 若是最后一折交叉验证
            X_valid, y_valid = X[val_start:], y[val_start:]  # 若不能整除，将多的case放在最后一折里
            X_train = X[0:val_start]
            y_train = y[0:val_start]

        return X_train, y_train, X_valid, y_valid

    def set_parameter(self, cv, model, number_class, sampling_rate,
                      random_seed, learning_rate, epoch, batch_size,
                      dropout, hiden_node, patient,
                      num_T, num_S, Lambda, hiden2):
        '''
        This is the function to set the parameters of training process and model
        All the settings will be saved into a NAME.txt file
        Input : cv --
                   The cross-validation type
                   Type = string
                   Default : Leave_one_session_out
                   Note : for different cross validation type, please add the
                          corresponding cross validation function. (e.g. self.Leave_one_session_out())

                model --
                   The model you want choose
                   Type = string
                   Default : TSception

                number_class --
                   The number of classes
                   Type = int
                   Default : 2

                sampling_rate --
                   The sampling rate of the EEG data
                   Type = int
                   Default : 256

                random_seed --
                   The random seed
                   Type : int
                   Default : 42

                learning_rate --
                   Learning rate
                   Type : flaot
                   Default : 0.001

                epoch --
                   Type : int
                   Default : 200

                batch_size --
                   The size of mini-batch
                   Type : int
                   Default : 128

                dropout --
                   dropout rate of the fully connected layers
                   Type : float
                   Default : 0.3

                hiden_node --
                   The number of hiden node in the fully connected layer
                   Type : int
                   Default : 128

                patient --
                   How many epoches the training process should wait for
                   It is used for the early-stopping
                   Type : int
                   Default : 4

                num_T --
                   The number of T kernels
                   Type : int
                   Default : 9

                num_S --
                   The number of S kernels
                   Type : int
                   Default : 6

                Lambda --
                   The L1 regulation coefficient in loss function
                   Type : float
                   Default : 1e-6

        '''
        self.model = model
        self.sampling_rate = sampling_rate
        # Parameters: Training process
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.num_epochs = epoch
        self.num_class = number_class
        self.batch_size = batch_size
        self.patient = patient
        self.Lambda = Lambda
        self.cv = cv
        # Parameters: Model
        self.dropout = dropout
        self.hiden_node = hiden_node
        self.T = num_T
        self.S = num_S
        self.hiden2 = hiden2
        # Save to log file for checking
        if cv == "Leave_one_subject_out":
            file = open("result_subject.txt", 'a')
        elif cv == "Leave_one_session_out":
            file = open("result_session.txt", 'a')
        elif cv == "K_fold":
            file = open("result_k_fold.txt", 'a')
        file.write("\n" + str(datetime.datetime.now()) +
                   "\nTrain:Parameter setting for " + str(self.model) +
                   "\n1)number_class:" + str(self.num_class) +
                   "\n2)random_seed:" + str(self.random_seed) +
                   "\n3)learning_rate:" + str(self.learning_rate) +
                   "\n4)num_epochs:" + str(self.num_epochs) +
                   "\n5)batch_size:" + str(self.batch_size) +
                   "\n6)dropout:" + str(self.dropout) +
                   "\n7)sampling_rate:" + str(self.sampling_rate) +
                   "\n8)hiden_node:" + str(self.hiden_node) +
                   "\n9)input_shape:" + str(self.input_shape) +
                   "\n10)patient:" + str(self.patient) +
                   "\n11)T:" + str(self.T) +
                   "\n12)S:" + str(self.S) +
                   "\n13)Lambda:" + str(self.Lambda) +
                   '\n')

        file.close()
    def k_fold(self, k, X_train, y_train):

        train_loss_sum, valid_loss_sum = 0, 0
        train_acc_sum, valid_acc_sum = 0, 0
        for i in range(k):
            print('*' * 25, '第', i + 1, '折', '*' * 25)
            train_data, train_label, val_data, val_label = self.get_kfold_data(k, i, X_train,
                                                                               y_train)  # 获取k折交叉验证的训练和验证数据
            desire_shape = [1, 14, 512]
            self.train_data_normal = RawEEGDataset(train_data, train_label, desire_shape)
            self.val_data_normal = RawEEGDataset(val_data, val_label, desire_shape)

            # 每份数据进行训练
            # train_loss, val_loss, train_acc, val_acc = self.traink(self.train_data_normal, self.val_data_normal,
            #                                                        learning_rate=self.learning_rate,
            #                                                        num_epochs=self.num_epochs, cv_type=self.cv)
            # train_loss, val_loss, train_acc, val_acc = self.objective(trial=100)
            study = optuna.create_study(direction="maximize")
            study.optimize(self.objective, n_trials=100)

            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))

            print("Best trial:")
            trial = study.best_trial

            print("  Value: ", trial.value)

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))

            # print('train_loss:{:.5f}, train_acc:{:.3f}%'.format(train_loss[-1], train_acc[-1]))
            # print('valid loss:{:.5f}, valid_acc:{:.3f}%\n'.format(val_loss[-1], val_acc[-1]))
            #
            # train_loss_sum += train_loss[-1]
            # valid_loss_sum += val_loss[-1]
            # train_acc_sum += train_acc[-1]
            # valid_acc_sum += val_acc[-1]

        print('\n', '#' * 10, '最终k折交叉验证结果', '#' * 10)

        print('average train loss:{:.4f}, average train accuracy:{:.3f}%'.format(train_loss_sum / k, train_acc_sum / k))
        print('average valid loss:{:.4f}, average valid accuracy:{:.3f}%'.format(valid_loss_sum / k, valid_acc_sum / k))


    def regulization(self, model, Lambda):
        w = torch.cat([x.view(-1) for x in model.parameters()])
        err = Lambda * torch.sum(torch.abs(w))
        return err

    def regulization2(self, model, Lambda):
        w = torch.cat([x.view(-1) for x in model.parameters()])
        err = Lambda * torch.sqrt_(torch.sum(w ** 2))
        return err

    def make_train_step(self, model, loss_fn, pairwise_loss, optimizer, scheduler):
        def train_step(x, y):
            model.train()
            yhat_pair, yhat = model(x)
            pred = yhat.max(1)[1]
            correct = (pred == y).sum()
            acc = correct.item() / len(pred)
            # L1 regularization
            optimizer.zero_grad()
            loss_r = self.regulization2(model, self.Lambda)
            # yhat is in one-hot representation;
            loss = loss_fn(yhat, y.long()) + loss_r
            loss.backward()
            # for name, parms in model.named_parameters():
            #      print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
            #           ' -->grad_value:', parms.grad)
            # scheduler.step()  # 余弦退火优化
            # loss2.backward()
            optimizer.step()
            # optimizer2.step( )

            # optimizer2.zero_grad()

            # y_train_plot = [i + 1 for i in y]
            # viz.scatter(X=yhat, Y=y_train_plot, win='Train_Scatter',
            #             opts=dict(marksize=1, legend=["negative", "neural",
            #                                           "positive"]),
            #             update='append')
            return loss.item(),acc

        return train_step

    def traink(self, train_data_normal, val_data_normal, learning_rate, num_epochs, cv_type):

        print('Avaliable device:' + str(torch.cuda.get_device_name(torch.cuda.current_device())))
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        # Train and validation loss
        losses = []
        accs = []

        Acc_val = []
        Loss_val = []
        val_losses = []
        val_acc = []

        test_losses = []
        test_acc = []
        Acc_test = []

        # hyper-parameter
        learning_rate = learning_rate
        num_epochs = num_epochs

        # model = CNN_LSTM_V2(self.num_class, inputsize=62 * 200, hiden=self.hiden_node, dropout_rate=self.dropout,
        #                     hiden2=self.hiden2, batch_size=self.batch_size).to(self.device)

    def objective(self, trial):
        print('Avaliable device:' + str(torch.cuda.get_device_name(torch.cuda.current_device())))

        torch.backends.cudnn.deterministic = True
        losses = []
        accs = []

        Acc_val = []
        Loss_val = []
        val_losses = []
        val_acc = []

        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        self.dropout = trial.suggest_uniform('dropout_rate', 0.5, 0.8)
        seed = trial.suggest_int('seed', 1, 200)
        self.loss_lambda = 1
        # lr = 8e-6
        # weight_decay = 1e-8
        # self.dropout = 0.2
        # lr = 1e-4
        # weight_decay =1e-4
        # self.dropout = 0.7
        # seed = 89
        torch.manual_seed(seed)
        model = inception_se_Final(self.num_class, inputsize=62 * 200, hiden=self.hiden_node, dropout_rate=self.dropout,
                                   hiden2=self.hiden2, batch_size=self.batch_size).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=0)
        # 等间隔调整学习率 StepLR   参数：
        # •optimizer: 神经网络训练中使用的优化器，如optimizer=torch.optim.SGD(...)
        # •step_size(int): 学习率下降间隔数，单位是epoch，而不是iteration.
        # •gamma(float): 学习率调整倍数，默认为0.1
        #
        # •last_epoch(int): 上一个epoch数，这个变量用来指示学习率是否需要调整。当last_epoch符合设定的间隔时，就会对学习率进行调整；当为-1时，学习率设置为初始值。
        loss_fn = nn.CrossEntropyLoss()
        pairwise_loss = Pairwise_Loss()
        if torch.cuda.is_available():
            model = model.to(self.device)
            loss_fn = loss_fn.to(self.device)
            self.loss_fn = loss_fn
            pairwise_loss = pairwise_loss.to(self.device)
        train_step = self.make_train_step(model, loss_fn, pairwise_loss, optimizer, scheduler)

        # load the data
        # Dataloader for training process
        # TODO: shuffle 改为TRUE
        train_data_loader = DataLoader(self.train_data_normal, batch_size=self.batch_size, shuffle=True,
                                       drop_last=True)
        val_data_loader = DataLoader(self.val_data_normal, batch_size=self.batch_size, shuffle=True, drop_last=True)


        ######## Training process ########
        Acc = []
        acc_max = 0
        patient = 0
        loss_min = 1000

        for epoch in range(self.num_epochs):

            loss_epoch = []

            acc_epoch = []
            for i, (x_batch, y_batch) in enumerate(train_data_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                loss,acc = train_step(x_batch, y_batch)
                loss_epoch.append(loss)
                acc_epoch.append(acc)
            losses.append(sum(loss_epoch) / len(loss_epoch))
            accs.append(sum(acc_epoch) / len(acc_epoch))
            loss_epoch = []
            acc_epoch = []
            print('Epoch [{}/{}], Loss: {:.4f},Acc: {:.4f}'
                  .format(epoch + 1, self.num_epochs, losses[-1],accs[-1]))

            ######## Validation process ########
            with torch.no_grad():
                correct2 = 0
                total = 0
                val_losses2 = []
                for x_val, y_val in val_data_loader:
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)

                    model.eval()
                    yhat_pair, yhat = model(x_val)

                    pred = yhat.max(1)[1]
                    correct = (pred == y_val).sum()
                    acc = correct.item() / len(pred)
                    val_loss = loss_fn(yhat, y_val.long())
                    val_losses.append(val_loss.item())
                    # val_losses2.append(val_loss2.item())
                    val_acc.append(acc)

                    # _, predicted = torch.max(yhat.data, 1)
                    # total += y_val.size(0)
                    # correct2 += (predicted == y_val).sum().item()

                Acc_val.append(sum(val_acc) / len(val_acc))
                Loss_val.append(sum(val_losses) / len(val_losses))
                # Loss2_val.append(sum(val_losses2) / len(val_losses2))
                trial.report(Acc_val[-1], epoch)
                print('Evaluation Loss:{:.4f}, Acc: {:.4f}'
                      .format(Loss_val[-1], Acc_val[-1]))
                val_losses = []
                val_losses2 = []
                val_acc = []
            ######## early stop ########

            Acc_es = Acc_val[-1]
            Loss_es = Loss_val[-1]
            if Loss_es < loss_min:
                loss_min = Loss_es
                patient = 0
                print('----Model saved!----')
                # torch.save(model.state_dict(), 'state_dict')
                file = open("result_acc.txt", 'r')
                lines = file.readlines()
                file = open("result_acc.txt", 'w')
                for line in lines:
                    acc = float(line[12:])
                    if Acc_es> acc:
                        # torch.save(model, 'max_model_2.pt')
                        # torch.save({'/state_dict': model.state_dict(), 'use_se': True}, 'BEST_checkpoint'+str(Acc_es)+'.tar',
                        #        _use_new_zipfile_serialization=False)
                        file.write("best acc is:" + str(Acc_es))
                        file.close()
                        torch.save(model.state_dict(),'BEST_checkpoint' + str(Acc_es) + '.tar')

                    else:
                        file.write(line)
                        file.close()
                        continue
                # torch.save(model, r'/max_model_2.pt')
            else:
                # torch.save(model, 'max_model_2.pt')
                # torch.save({'state_dict':model.state_dict(),'use_se': True},'BEST_checkpoint.tar',_use_new_zipfile_serialization=False)
                # def convert_model(model, input=torch.tensor(torch.rand(size=(1, 3, 112, 112)))):
                # model2 = torch.jit.trace(model,torch.tensor(torch.rand(size=(1, 1, 14, 256))).cuda())
                # torch.jit.save(model2, 'model_test.tjm')
                patient += 1
            if patient > self.patient:
                print('----Early stopping----')
                self.stopepoch = epoch
                break
        scheduler.step()

        ## 画图

        x_trainloss = range(0, self.stopepoch + 1)
        x_trainacc = range(0, self.stopepoch + 1)
        y_trainloss = losses
        y_trainacc = accs
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(x_trainloss, y_trainloss, 'o-')
        plt.title('Train loss vs. epoches')
        plt.ylabel('Train loss')
        plt.subplot(2, 1, 2)
        plt.plot(x_trainacc, y_trainacc, '.-')
        plt.xlabel('Train Acc vs. epoches')
        plt.ylabel('Train Acc')
        plt.savefig(r"Train accuracy_loss_" + str(datetime.datetime.now().strftime("%Y-%m-%d")) + '.jpg')
        # plt.show()
        x_valloss = range(0, self.stopepoch + 1)
        x_valacc = range(0, self.stopepoch + 1)
        y_valloss = Loss_val
        y_valacc = Acc_val
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(x_valloss, y_valloss, 'o-')
        plt.title('valloss vs. epoches')
        plt.ylabel('val loss')
        plt.subplot(2, 1, 2)
        plt.plot(x_valacc, y_valacc, '.-')
        plt.xlabel('val Acc vs. epoches')
        plt.ylabel('val Acc')
        plt.savefig(r"Val accuracy_loss_" + str(datetime.datetime.now().strftime("%Y-%m-%d")) + '.jpg')
        # save the loss(acc) for plotting the loss(acc) curve
        save_path = Path(os.getcwd())
        cv_type = "leave_one_session_out"
        if cv_type == "leave_one_session_out":
            filename_callback = save_path / Path('_history_version2.hdf')
            save_history = h5py.File(filename_callback, 'w')
            save_history['acc'] = accs
            save_history['loss'] = losses
            save_history.close()
        # file = open("result_k_fold.txt", 'a')
        # file.write("\n" + str(datetime.datetime.now()) +
        #            "\n ** 网络结构 **： " + str(self.model) +
        #            "\n ** 验证集准确率 **：" + str(Acc_val[-1]) +
        #            "\n ** 测试集准确率 **:" + str(Acc_test) +
        #            "\n ** 程序单次测试时间 **" + str(datetime.datetime.now() - time_start) +
        #            "\n ** 实验人： 龚鸣  **"
        #            '\n')
        # self.losses = losses
        # self.Loss_val = Loss_val
        # self.accs = accs
        # self.ACC_val = Acc_val
        best_acc = 0
        best_acc = max(best_acc, Acc_val[-1])
        return best_acc


def test(device,test_data,test_label):
    ######## test process ########
    model = inception_se_Final(train.num_class, inputsize=62 * 200, hiden=train.hiden_node, dropout_rate=train.dropout,
                               hiden2=train.hiden2, batch_size=train.batch_size).to(train.device)
    test_losses = []
    test_acc = []
    desire_shape = [1, 14, 512]
    loss_fn = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.to(device)
        loss_fn = loss_fn.to(device)
    test_data_normal = RawEEGDataset(test_data, test_label, desire_shape)
    test_data_loader = DataLoader(test_data_normal, batch_size=train.batch_size, shuffle=True,
                                  drop_last=True)
    print(os.getcwd())
    file_name = new_file(os.getcwd())
    if file_name.endswith('.tar'):
        torch.backends.cudnn.deterministic = True
        # model = torch.load('max_model_2.pt')
        state_dict = torch.load(file_name)
        model.load_state_dict(state_dict=state_dict)
        model.to(device)
        # model_dict = torch.load(file_name)
        # model.load_state_dict(model_dict)


    with torch.no_grad():
        for x_test, y_test in test_data_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            model.eval()
            yhat_pair, yhat = model(x_test)
            pred = yhat.max(1)[1]
            correct = (pred == y_test).sum()
            acc = correct.item() / len(pred)
            test_loss = loss_fn(yhat, y_test.long())
            test_losses.append(test_loss.item())
            test_acc.append(acc)

        print('Test Loss:{:.4f}, Acc: {:.4f}'
              .format(sum(test_losses) / len(test_losses), sum(test_acc) / len(test_acc)))
        print('precision score is ',
              precision_score(y_test.cpu(), pred.cpu(), average="micro"))  # 输出多分类问题的精准率的大小（需要设定average参数）
        print('recall score is ', recall_score(y_test.cpu(), pred.cpu(), average="micro"))  # 输出多分类问题的召回率
        print('confusion martix is\n', confusion_matrix(y_test.cpu(), pred.cpu()))  # 输出混淆矩阵

if __name__ == "__main__":
    train = TrainModel()
    os.chdir('split_data')
    for i in os.listdir('.'):
        if not i.endswith('.hdf'):
            continue
        print(os.getcwd()+'\\'+i)
        train.load_data(os.getcwd() +'\\' +i)
    os.chdir(os.path.pardir)
    os.makedirs('result', exist_ok=True)
    os.chdir('result')
    # train.set_parameter(cv='K_fold',
    #                     model='CNN_LSTM_V2',
    #                     number_class=3,
    #                     sampling_rate=200,
    #                     random_seed=42,
    #                     learning_rate=0.0001,
    #                     epoch=1500,
    #                     batch_size=128,
    #                     dropout=0.6,
    #                     hiden_node=64,
    #                     hiden2=32,
    #                     patient=25,
    #                     num_T=9,
    #                     num_S=6,
    #                     Lambda=0.0004) cross entropy loss
    train.set_parameter(cv='K_fold',
                        model='inception_se_Final',
                        number_class=3,
                        sampling_rate=200,
                        random_seed=42,
                        learning_rate=0.00008,
                        epoch=1500,
                        batch_size=512,
                        dropout=0.1,
                        hiden_node=32,
                        hiden2=32,
                        patient=25,
                        num_T=9,
                        num_S=6,
                        Lambda=0.001)

    train.k_fold(k=10, X_train=train.data, y_train=train.label)
    test(train.device,train.test_data,train.test_label)
    # train.train(train_data=train.train_data,train_label=train.train_label,test_data=train.test_data,test_label=train.test_label,learning_rate=train.learning_rate,num_epochs=train.num_epochs,cv_type=train.cv)
