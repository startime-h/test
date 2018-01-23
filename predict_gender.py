import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from keras import optimizers
from keras.layers import Dense
from keras.layers import Input, Flatten
from keras.models import Model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from load_data import get_sample_csv_v

np.random.seed(1001)
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


def build_v_model():
    main_input = Input(shape=(1, 88), name='main_input')

    x = Dense(64, activation='tanh', kernel_regularizer='l2')(main_input)
    x = Dense(64, activation='tanh', kernel_regularizer='l2')(x)
    x = Dense(64, activation='tanh', kernel_regularizer='l2')(x)

    x = Flatten()(x)
    # And finally we add the main logistic regression layer
    y_v_label = Dense(1, activation='softplus', name='y_v_label')(x)

    model = Model(inputs=main_input, outputs=[y_v_label])

    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

    model.summary()
    return model


nb_epoch = 100
batch_size = 100

# 1.buildModel
model = build_v_model()

# 2.getFeature
X, y_v, y_a, y_d, y_gender, y_emotion = get_sample_csv_v(
    ids=None, path_to_features='F:\\Session1\\dialog\\feature\\', take_all=True)
# 3.将数据分测试集、验证集, 随机拆分数据, 结果存在随机性

idxs_train, idxs_test = train_test_split(range(X.shape[0]), test_size=0.2)

X_train, X_test = X[idxs_train], X[idxs_test]
shape0 = X_train.shape[0]
shape1 = X_train.shape[1]
shape2 = X_train.shape[2]

# 3.1.数据预处理
reshapeX_train = preprocessing.scale(np.reshape(X_train, (shape0, shape2)))
reshapeX_test = preprocessing.scale(np.reshape(
    X_test, (X_test.shape[0], X_test.shape[2])))

X_train = reshapeX_train.reshape(X_train.shape)
X_test = reshapeX_test.reshape(X_test.shape)

y_train_v, y_test_v = y_v[idxs_train], y_v[idxs_test]

epoch_time1 = time.time()

# 4.Trains the model for a fixed number of epochs
hist = model.fit(X_train, y_train_v,
                 batch_size=batch_size, epochs=nb_epoch, verbose=1,
                 validation_data=(X_test, y_test_v))

# 5.Returns the loss value & metrics values for the model in test mode.
loss, accuracy = model.evaluate(X_test, y_test_v)

print("test loss:", loss)
print("test accuracy:", accuracy)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper right")
plt.show()
