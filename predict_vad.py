import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from load_data import *
from models import build_model_VAD_M1, build_model_VAD_M2

np.random.seed(1337)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

nb_epoch = 300
batch_size = 32
model = build_model_VAD_M2()

# 2.getFeature
X, y_v, y_a, y_d, y_gender, y_emotion = get_sample_csv_v(
    ids=None, path_to_features='C:\\BaiduYunDownload\\features-6373\\features2\\', take_all=True)

# 3.将数据分测试集、验证集
idxs_train, idxs_test = train_test_split(range(X.shape[0]), test_size=0.2)

X_train, X_test = X[idxs_train], X[idxs_test]
y_v_train, y_v_test = y_v[idxs_train], y_v[idxs_test]
y_a_train, y_a_test = y_a[idxs_train], y_v[idxs_test]
y_d_train, y_d_test = y_d[idxs_train], y_d[idxs_test]

shape0 = X_train.shape[0]
shape1 = X_train.shape[1]
shape2 = X_train.shape[2]

# 3.1 standardize the data directly
reshapeX_train = preprocessing.scale(np.reshape(X_train, (shape0, shape2)))
reshapeX_test = preprocessing.scale(
    np.reshape(X_test, (X_test.shape[0], X_test.shape[2])))

norX_train = preprocessing.normalize(reshapeX_train, norm='l2')
norX_test = preprocessing.normalize(reshapeX_test, norm='l2')

X_train = norX_train.reshape(X_train.shape)
X_test = norX_test.reshape(X_test.shape)

hist = model.fit(
    X_train, [y_v_train, y_a_train, y_d_train],
    batch_size=batch_size,
    epochs=nb_epoch,
    validation_split=0.2)  # 此处切分训练集用来验证, 观察测试

# 评估效果

score = model.evaluate(X_test, [y_v_test, y_a_test, y_d_test], batch_size=256)

# 当模型是多输出时 score为一个<class 'list'>: [1.0049264339967208, 0.68758998740803112,
# 1.2882261137528852, 0.47236362554810263, 0.17454545465382662, 0.14545454548163847,
#  0.20000000065023249], 现在已知的有第一个为val_loss之和, 随后几个为val_v_loss, val_a_loss, val_d_loss
# 再后面暂时还不清楚(正在查阅相关资料), 通过观察hist.history, 现在的效果好多了

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.title("v, a, d prediction")

plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper right")
plt.show()
