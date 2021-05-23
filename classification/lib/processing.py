import csv
import os

csv_file = open(r'D:\python\emotion_classification\classification\lib\headerRemoved\demo_marker2021-05-13_EPOCX_128237_2021.05.13T15.34.34+08.00.md.pm.bp.csv')
csv_lines = csv.DictReader(csv_file)
name_list = ['EEG.AF3','EEG.F7','EEG.F3','EEG.FC5','EEG.T7','EEG.P7','EEG.O1','EEG.O2','EEG.P8','EEG.T8','EEG.FC6','EEG.F4','EEG.F8','EEG.AF4','MarkerIndex','MarkerType']
output = 'data.csv'
csv_out = open(output,'wt')
csv_writer = csv.writer(csv_out)
csv_writer.writerow(name_list)
num_start = 0
num_end = 0
dict_key = {}
values = []

for i, csv_line in enumerate(csv_lines):
    value_1, value_2, value_3, value_4, value_5, value_6, value_7, value_8, value_9, value_10, value_11, value_12, value_13, value_14, key, type = \
        csv_line['EEG.AF3'], csv_line['EEG.F7'], csv_line['EEG.F3'], csv_line['EEG.FC5'], csv_line['EEG.T7'], \
        csv_line['EEG.P7'], csv_line['EEG.O1'], csv_line['EEG.O2'], csv_line['EEG.P8'], csv_line['EEG.T8'], \
        csv_line['EEG.FC6'], csv_line['EEG.F4'], csv_line['EEG.F8'], csv_line['EEG.AF4'], csv_line['MarkerIndex'], \
        csv_line['MarkerType']
    values = [value_1, value_2, value_3, value_4, value_5, value_6, value_7, value_8, value_9, value_10,
              value_11, value_12, value_13, value_14, key, type]
    if key:
        dict_key[int(key)] = i
    for index, value in dict_key.items():
        if index % 3 == 2:
            num_start = value
            print("num_start", num_start)
        elif index % 3 == 0:
            num_end = value
            print("num_end", num_end)
    if i in range(num_start,num_end):
        csv_writer.writerow(values)
























