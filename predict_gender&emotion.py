import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from load_data import get_sample_csv_v, to_category
from refer import build_model_e_g
from keras.utils import np_utils
# gender level
nb_epoch = 50
batch_size = 10

available_emotions = np.array(['ang', 'exc', 'neu', 'sad', 'fru', 'hap', 'fea', 'dis'])
gender_values = np.array(['F', 'M'])

model_g = build_model_e_g()

# 2.getFeature
X, y_v, y_a, y_d, y_gender, y_emotion = get_sample_csv_v(
    ids=None, path_to_features='F:\\Session1\\dialog\\feature\\Ses01F_impro01\\', take_all=True)

# X, _ = pad_sequence_into_array(X)

y_gender_1 = to_category(y_gender, gender_values)
y_emotion_1 = to_category(y_emotion, available_emotions)

# 3.将数据分测试集、验证集
idxs_train, idxs_test = train_test_split(range(X.shape[0]), test_size=0.2)
# X, _ = pad_sequence_into_array(X)

print("------")
print(y_emotion)

X_train, X_test = X[idxs_train], X[idxs_test]
y_train_emotion, y_test_emotion = y_emotion_1[
                                      idxs_train], y_emotion_1[idxs_test]
y_train_gender, y_test_gender = y_gender_1[idxs_train], y_gender_1[idxs_test]

shape0 = X_train.shape[0]
shape1 = X_train.shape[1]
shape2 = X_train.shape[2]
# 3.1 standardize the data directly
reshapeX_train = preprocessing.scale(np.reshape(X_train, (shape0, shape2)))
reshapeX_test = preprocessing.scale(np.reshape(
    X_test, (X_test.shape[0], X_test.shape[2])))

norX_train = preprocessing.normalize(reshapeX_train, norm='l2')
norX_test = preprocessing.normalize(reshapeX_test, norm='l2')

X_train = norX_train.reshape(X_train.shape)
X_test = norX_test.reshape(X_test.shape)


epoch_time1 = time.time()
# early_stopping = EarlyStopping(monitor='val_loss',patience = 2)
hist_g = model_g.fit(X_train, [y_train_emotion, y_train_gender],
                     batch_size=batch_size, epochs=nb_epoch, verbose=1,
                     validation_data=(X_test, [y_test_emotion, y_test_gender]))

# 5.Returns the loss value & metrics values for the model in test mode.
loss, accuracy = model_g.evaluate(X_test, [y_test_emotion, y_test_gender],
                           batch_size=batch_size)

epoch_time2 = time.time()
print('time last: %d', (epoch_time2 - epoch_time1) / 60)


# summarize history for loss
plt.plot(hist_g.history['loss'])
plt.plot(hist_g.history['val_loss'])

plt.title("relu level3 batch 100 category4 sgd")
# plt.title("category4 loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper right")
plt.show()
