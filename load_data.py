import csv
import os

import numpy as np


def get_sample_csv_v(ids, path_to_features, take_all=False):
    if take_all:
        orig = {}
        files = os.listdir(path_to_features)  # 列出文件夹下所有文件, files是一个list
        for i in files:
            if '.' not in i:
                file = os.listdir(path_to_features + i + '/')
                file_list = np.sort([f[:-4] for f in file if f.endswith(".csv")])
                orig[i] = file_list
        # 列表迭代器过滤出csv文件
    orig_x = []
    y_vs = []
    y_as = []
    y_ds = []
    y_genders = []
    y_emotions = []
    for key, value in orig.items():
        for i in value:
            x, y_v, y_a, y_d, y_gender, y_emotion = load_sample_csv(
                path_to_features + key + '/' + i + '.csv', i)
            if len(x) > 0:
                orig_x.append(np.array(x, dtype=float))
                y_vs.append(y_v)
                y_as.append(y_a)
                y_ds.append(y_d)
                y_genders.append(y_gender)
                y_emotions.append(y_emotion)
    orig_x = np.array(orig_x)
    y_vs = np.array(y_vs)
    y_as = np.array(y_as)
    y_ds = np.array(y_ds)
    y_genders = np.array(y_genders)
    y_emotions = np.array(y_emotions)
    return orig_x, y_vs, y_as, y_ds, y_genders, y_emotions


def get_sample(path):
    files = os.listdir(path)
    file = [f[:-4] for f in files if f.endswith(".csv")]
    orig_x = []
    y_vs = []
    y_as = []
    y_ds = []
    y_emotions = []
    y_gender = []
    for f in file:
        x, v, a, d, e, g = load_sample_csv(path + "\\" + f + ".csv", f)
        orig_x.append(x)
        y_vs.append(v)
        y_as.append(a)
        y_ds.append(d)
        y_emotions.append(e)
        y_gender.append(g)
    return orig_x, y_vs, y_as, y_ds, y_gender, y_emotions


def load_sample_csv(feature_file_full_name, filename):
    with open(feature_file_full_name, 'r') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        x = []
        y_vs = []
        y_as = []
        y_ds = []
        y_genders = []
        y_emotions = []
        y_v, y_a, y_d, y_gender, y_emotion = load_label_y(filename)
        for row in rows:
            if len(row) > 1:
                length = len(row)
                x.append(row[1:length - 1])
                # 获得数据信息
                y_vs.append(y_v)
                y_as.append(y_a)
                y_ds.append(y_d)
                y_genders.append(y_gender)
                y_emotions.append(y_emotion)
    return x, y_vs, y_as, y_ds, y_genders, y_emotions


def load_label_y(feature_file_name):
    sub_str_array = feature_file_name.split('@')
    v_a_d_gender = sub_str_array[0].split('#')
    vad = v_a_d_gender[2].split('_')
    gender = v_a_d_gender[1]
    emotion = sub_str_array[1]  # fru, ang
    v = vad[0]  # 2.0000
    a = vad[1]  # 3.5000
    d = vad[2]  # 3.5000
    return v, a, d, gender, emotion


def to_category(y, values):
    y_shape = sorted(y.shape)
    v_shape = sorted(values.shape)
    y = y.reshape(y_shape[1], y_shape[0])
    values = values.reshape(1, v_shape[0])
    # python中的广播
    out = np.array(y == values)
    ones = np.ones(out.shape)
    return (out * ones).reshape(y_shape[1], v_shape[0])
