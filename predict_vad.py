import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from load_data import *
from refer import build_model_VAD_M1
import numpy as np
np.random.seed(1337)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
nb_epoch = 100
batch_size = 256
model = build_model_VAD_M1()

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

hist_g = model.fit(
    X_train, [y_v_train, y_a_train, y_d_train],
    batch_size=batch_size,
    epochs=nb_epoch,
    validation_data=(X_test, [y_v_test, y_a_test, y_d_test]))

score = model.evaluate(X_test, [y_v_test, y_a_test, y_d_test], batch_size=256)

# print("\nscore:", score)
# print("\ntest loss: ", score[0])
# print("test accuracy: ", score[1])
# print("history:", hist_g.history)
plt.plot(hist_g.history['loss'])
plt.plot(hist_g.history['val_loss'])

plt.title("v, a, d prediction")

plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper right")
plt.show()
