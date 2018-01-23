import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from load_data import get_sample_csv_v, to_category
from models import predict_All

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(1001)

available_emotions = np.array(['ang', 'exc', 'neu', 'sad', 'fru', 'hap', 'fea', 'dis'])

np_epoch = 50

batch_size = 200

X, y_v, y_a, y_d, y_gender, y_emotion = get_sample_csv_v(
    ids=None, path_to_features='C:\\BaiduYunDownload\\features-6373\\features2\\', take_all=True)

idxs_train, idxs_test = train_test_split(range(X.shape[0]), test_size=0.2)

y_emotion = to_category(y_emotion, available_emotions)

X_train, X_test = X[idxs_train], X[idxs_test]
y_train_emotion, y_test_emotion = y_emotion[idxs_train], y_emotion[idxs_test]
y_v_train, y_v_test = y_v[idxs_train], y_v[idxs_test]
y_a_train, y_a_test = y_a[idxs_train], y_v[idxs_test]
y_d_train, y_d_test = y_d[idxs_train], y_d[idxs_test]

shape0 = X_train.shape[0]
shape1 = X_train.shape[1]
shape2 = X_train.shape[2]

# 3.1 standardize the data directly
reshapeX_train = preprocessing.scale(np.reshape(X_train, (shape0, shape2)))
reshapeX_test = preprocessing.scale(np.reshape(X_test, (X_test.shape[0], X_test.shape[2])))

norX_train = preprocessing.normalize(reshapeX_train, norm='l2')
norX_test = preprocessing.normalize(reshapeX_test, norm='l2')

X_train = norX_train.reshape(X_train.shape)
X_test = norX_test.reshape(X_test.shape)

model = predict_All()

hist = model.fit(X_train, [y_train_emotion, y_v_train, y_a_train, y_d_train], batch_size=batch_size, epochs=np_epoch,
                 validation_data=(X_test, [y_test_emotion, y_v_test, y_a_test, y_d_test]))

score = model.evaluate(X_test, [y_test_emotion, y_v_test, y_a_test, y_d_test])


plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.show()
