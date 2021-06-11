import csv, os
import numpy as np
import h5py
from pathlib import Path
import json
import mne
import math
# Loop through every file in the current working directory.
os.makedirs('orginal_data', exist_ok=True)
os.chdir('orginal_data')

def generate_label(video_start , video_end):
    for subject_name in os.listdir():
        os.chdir(subject_name)
        for csvFilename in os.listdir('.'):
            if not csvFilename.endswith('.json'):
                continue
            count = 0
            with open(csvFilename, 'r', encoding='utf-8') as f:
                markers = json.load(f)
                marker = markers['Markers']
                label = []
                label_type = {'Positive': 2, 'Neural': 1, 'Negative': 0}
                for i in range(len(marker) - 1):
                    if i % 3 == 0:
                        label_name = marker[i + 1]['value']
                        for key in label_type.keys():
                            if key in label_name:
                                label_array = np.ones((video_end[count]-video_start[count]))
                                label_array = label_array * label_type[key]
                                label.append(label_array)
                                count +=1
                label = np.hstack(label)
                print(os.getcwd())
                os.chdir(os.path.pardir)
                print(os.getcwd())
                return label
os.makedirs('headerRemoved', exist_ok=True)
for subject_name in os.listdir():
    os.chdir(subject_name)
    if subject_name =='headerRemoved':
        os.chdir(os.path.pardir)
        continue
    for csvFilename in os.listdir('.'):
        if not csvFilename.endswith('.csv'):
            continue  # skip non-csv files
        # print('Removing header from ' + csvFilename + '...')

        # Read the CSV file in (skipping first row).
        csvRows = []
        csvFileObj = open(csvFilename)
        readerObj = csv.reader(csvFileObj)
        for row in readerObj:
            if readerObj.line_num == 1:
                continue  # skip first row
            csvRows.append(row)
        csvFileObj.close()
        os.chdir(os.path.pardir)
        # Write out the CSV file.
        csvFileObj = open(os.path.join('headerRemoved', csvFilename), 'w',
                          newline='')
        csvWriter = csv.writer(csvFileObj)
        for row in csvRows:
            csvWriter.writerow(row)
        csvFileObj.close()

def new_file(test_dir):
    #列举test_dir目录下的所有文件（名），结果以列表形式返回。
    lists=os.listdir(test_dir)
    #sort按key的关键字进行升序排序，lambda的入参fn为lists列表的元素，获取文件的最后修改时间，所以最终以文件时间从小到大排序
    #最后对lists元素，按文件修改时间大小从小到大排序。
    lists.sort(key=lambda fn:os.path.getmtime(test_dir+'\\'+fn))
    #获取最新文件的绝对路径，列表中最后一个值,文件夹+文件名
    file_path=os.path.join(test_dir,lists[-1])
    return file_path

def split_data(data,label,segment_length=2, overlap=0.95, sampling_rate=256, save=True ):
    '''
    用来切分输入数据
    segment_length:时间窗长度，单位s
    overlap：负采样率，以小数形式表示
    '''
    data_shape = data.shape
    label_shape = label.shape
    data_step = int(segment_length * sampling_rate * (1 - overlap))
    data_segment = sampling_rate * segment_length
    data_split = []
    label_split = []
    number_segment = int((data_shape[1] - data_segment) // (data_step)) + 1
    for i in range(number_segment):
        data_split.append(data[:,(i * data_step):(i * data_step + data_segment)])
        # label_split.append(label)
        label_split.append(label[(i * data_step)])
    data_split_array = np.stack(data_split, axis=1)
    label_split_array = np.stack(label_split, axis=0)
    print("The data and label are splited: Data shape:" + str(data_split_array.shape) + " Label:" + str(
        label_split_array.shape))
    # data : trial x segment x datapoint x 1
    # label : trial x datapoint
    data = data_split_array
    label = label_split_array
    return data,label
import pandas as pd
file_name = new_file(os.path.join('headerRemoved'))  # 去掉EPOC X CSV文件首行的不需要讯息
data_csv = pd.read_csv(file_name,low_memory=False)
data = pd.DataFrame(data_csv)
name_list_orginal = ['Timestamp','EEG.Counter','EEG.Interpolated','EEG.AF3','EEG.F7','EEG.F3','EEG.FC5','EEG.T7','EEG.P7','EEG.O1','EEG.O2','EEG.P8','EEG.T8','EEG.FC6','EEG.F4','EEG.F8','EEG.AF4','EEG.RawCq','EEG.Battery','EEG.BatteryPercent','MarkerIndex','MarkerType','MarkerValueInt','EEG.MarkerHardware','CQ.AF3','CQ.F7','CQ.F3','CQ.FC5','CQ.T7','CQ.P7','CQ.O1','CQ.O2','CQ.P8','CQ.T8','CQ.FC6','CQ.F4','CQ.F8','CQ.AF4','CQ.Overall','EQ.SampleRateQuality','EQ.OVERALL','EQ.AF3','EQ.F7','EQ.F3','EQ.FC5','EQ.T7','EQ.P7','EQ.O1','EQ.O2','EQ.P8','EQ.T8','EQ.FC6','EQ.F4','EQ.F8','EQ.AF4','MOT.CounterMems','MOT.InterpolatedMems','MOT.Q0','MOT.Q1','MOT.Q2','MOT.Q3','MOT.AccX','MOT.AccY','MOT.AccZ','MOT.MagX','MOT.MagY','MOT.MagZ','PM.Engagement.IsActive','PM.Engagement.Scaled','PM.Engagement.Raw','PM.Engagement.Min','PM.Engagement.Max','PM.Excitement.IsActive','PM.Excitement.Scaled','PM.Excitement.Raw','PM.Excitement.Min','PM.Excitement.Max','PM.LongTermExcitement','PM.Stress.IsActive','PM.Stress.Scaled','PM.Stress.Raw','PM.Stress.Min','PM.Stress.Max','PM.Relaxation.IsActive','PM.Relaxation.Scaled','PM.Relaxation.Raw','PM.Relaxation.Min','PM.Relaxation.Max','PM.Interest.IsActive','PM.Interest.Scaled','PM.Interest.Raw','PM.Interest.Min','PM.Interest.Max','PM.Focus.IsActive','PM.Focus.Scaled','PM.Focus.Raw','PM.Focus.Min','PM.Focus.Max','POW.AF3.Theta','POW.AF3.Alpha','POW.AF3.BetaL','POW.AF3.BetaH','POW.AF3.Gamma','POW.F7.Theta','POW.F7.Alpha','POW.F7.BetaL','POW.F7.BetaH','POW.F7.Gamma','POW.F3.Theta','POW.F3.Alpha','POW.F3.BetaL','POW.F3.BetaH','POW.F3.Gamma','POW.FC5.Theta','POW.FC5.Alpha','POW.FC5.BetaL','POW.FC5.BetaH','POW.FC5.Gamma','POW.T7.Theta','POW.T7.Alpha','POW.T7.BetaL','POW.T7.BetaH','POW.T7.Gamma','POW.P7.Theta','POW.P7.Alpha','POW.P7.BetaL','POW.P7.BetaH','POW.P7.Gamma','POW.O1.Theta','POW.O1.Alpha','POW.O1.BetaL','POW.O1.BetaH','POW.O1.Gamma','POW.O2.Theta','POW.O2.Alpha','POW.O2.BetaL','POW.O2.BetaH','POW.O2.Gamma','POW.P8.Theta','POW.P8.Alpha','POW.P8.BetaL','POW.P8.BetaH','POW.P8.Gamma','POW.T8.Theta','POW.T8.Alpha','POW.T8.BetaL','POW.T8.BetaH','POW.T8.Gamma','POW.FC6.Theta','POW.FC6.Alpha','POW.FC6.BetaL','POW.FC6.BetaH','POW.FC6.Gamma','POW.F4.Theta','POW.F4.Alpha','POW.F4.BetaL','POW.F4.BetaH','POW.F4.Gamma','POW.F8.Theta','POW.F8.Alpha','POW.F8.BetaL','POW.F8.BetaH','POW.F8.Gamma','POW.AF4.Theta','POW.AF4.Alpha','POW.AF4.BetaL','POW.AF4.BetaH','POW.AF4.Gamma']
name_list = ['EEG.AF3','EEG.F7','EEG.F3','EEG.FC5','EEG.T7','EEG.P7','EEG.O1','EEG.O2','EEG.P8','EEG.T8','EEG.FC6','EEG.F4','EEG.F8','EEG.AF4','MarkerIndex','MarkerType']

name_del = [i for i in name_list_orginal if i not in name_list]
data = data.drop(columns=name_del,axis=1)  # 保留所需EEG通道和marker通道信息
marker_split = data.loc[data["MarkerType"] == 1].index  # 按照markertype 中的信息对数据进行截断，取实验中刺激阶段数据拼接
video_start = []
video_end = []
for i in range(len(marker_split)-1):
    if i%3==0:
        video_start.append(marker_split[i+1])
        video_end.append(marker_split[i+2])
label = generate_label(video_start,video_end)
for i in range(len(video_start)):
    if i == 0:
        data_processed = data[video_start[i]:video_end[i]]
        data_processed = np.array(data_processed)
    else:
        data_split = data[video_start[i]:video_end[i]]
        data_split = np.array(data_split)
        data_processed = np.concatenate([data_processed,data_split],axis=0)
data_processed = np.delete(data_processed,[14,15],axis=1) # 删除marker对应的列
data_processed = np.swapaxes(data_processed,0,1)
'''
此部分意图用mne对于脑电信号进行滤波陷波处理
'''
ch_name = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']

info = mne.create_info(
    ch_names=ch_name,
    ch_types=['eeg', 'eeg', 'eeg', 'eeg','eeg', 'eeg', 'eeg', 'eeg','eeg', 'eeg', 'eeg', 'eeg','eeg', 'eeg'],
    sfreq=256
)
raw = mne.io.RawArray(data_processed, info)
montage = mne.channels.make_standard_montage('standard_1020')
# print([name for name in raw.info['ch_names']])
on_missing_montage = []
for names in montage.ch_names:
    if names not in raw.info['ch_names']:
        on_missing_montage.append(names)
# print(montage.ch_names)
# print(on_missing_montage)
raw.set_montage('standard_1020')
info.set_montage(montage)
# raw.plot(n_channels=len(raw.ch_names),scalings = 'auto',)

raw.filter(1, 50, fir_design='firwin')
raw.load_data().notch_filter(50,trans_bandwidth=1.0,fir_window='hamming')
# raw.plot_psd()
import time
start_ica_time = time.time()
ica = mne.preprocessing.ICA(n_components=14,random_state=97, max_iter=800)
ica.fit(raw)
org_raw = raw.copy()
raw.load_data()
ica.exclude = [0,1]
raw = ica.apply(raw)
# ica.plot_sources(raw)
# ica.save('test_ica')
data = ica.get_sources(inst=raw)
data = data[2:][0]
# raw.plot(n_channels=len(raw.ch_names),scalings = 'auto',show_scrollbars=False)
# # org_raw.plot(n_channels=len(raw.ch_names),scalings = 'auto',show_scrollbars=False)
# ica.plot_components()
# ica.plot_properties(raw,picks=ica.exclude) # RuntimeError: No digitization points found.不会解决，画不出ica的图  解决方法，TMD把设置montage通道电极的代码放在ICA之前！！！
data = np.array(data)
'''
将14*采样点，按照采样率256hz进行截断，区分trial
'''
length = len(data_processed[1])
data,label = split_data(data,label)
# pos = 0
# data_list = []
# label_list = []
# while pos + 256 <= length:
#     data_list.append(np.asarray(data_processed[:, pos:pos + 256]))
#     label_list.append(label[pos])  # 截取片段对应的 label，-1, 0, 1
#     pos += 256
data = np.array(data)
data = data.swapaxes(0,1)
label = np.array(label)
index = np.arange(data.shape[0])
np.random.seed(22)
np.random.shuffle(index)
label = label[index]
data = data[index,:,:]
split_num = int(data.shape[0] * 0.9)
test_data = data[split_num:,:,:]  # 从打乱的数据中取百分之二十作为test
test_label = label[split_num:]
data = data[0:split_num,:,:]
label = label[0:split_num]
os.chdir(os.path.pardir)
os.makedirs('split_data', exist_ok=True)
os.chdir('split_data')
save_path = Path(os.getcwd())
filename_data = save_path / Path('data_split.hdf')
save_data = h5py.File(filename_data, 'w')
save_data['data'] = data
save_data['label'] = label
save_data['test_data'] = test_data
save_data['test_label'] = test_label
save_data.close()
