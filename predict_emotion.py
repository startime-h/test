import time
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from load_data import get_sample_csv_v, to_category
from refer import build_model_emotion, build_model_emotion1

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(32)

nb_epoch = 100
batch_size = 50

available_emotions = np.array(['ang', 'exc', 'neu', 'sad', 'fru', 'hap', 'fea', 'dis'])
# available_emotions = np.array(['ang', 'exc', 'neu', 'sad'])

model_g = build_model_emotion1()

# 2.getFeature
X, y_v, y_a, y_d, y_gender, y_emotion = get_sample_csv_v(
    ids=None, path_to_features='C:\\BaiduYunDownload\\features-6373\\features2\\', take_all=True)


# 3.将数据分测试集、验证集
idxs_train, idxs_test = train_test_split(range(X.shape[0]), test_size=0.2)

y_emotion = to_category(y_emotion, available_emotions)

X_train, X_test = X[idxs_train], X[idxs_test]
y_train_emotion, y_test_emotion = y_emotion[idxs_train], y_emotion[idxs_test]

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


early_stopping = EarlyStopping(monitor='val_loss', patience=2)

hist_g = model_g.fit(X_train, y_train_emotion,
                     batch_size=batch_size, epochs=nb_epoch, verbose=1,
                     validation_data=(X_test, y_test_emotion), callbacks=[early_stopping])

# 5.Returns the loss value & metrics values for the model in test mode.
loss, accuracy = model_g.evaluate(X_test, y_test_emotion)


model_g.save('emotion_predict.h5')
print("test loss: ", loss)
print("test accuracy: ", accuracy)
# summarize history for loss
plt.plot(hist_g.history['loss'])
plt.plot(hist_g.history['val_loss'])

plt.title("只包含softmax激活层模型", fontproperties='SimHei', fontsize=15)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper right")
plt.show()