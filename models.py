from keras import Input
from keras import optimizers
from keras.layers import Dense, Flatten, concatenate, Dropout
from keras.models import Model

from metrics import concordance_cc


def build_model_emotion1():
    """
    构建模型, 三个隐藏层, 使用的是sigmoid激活函数
    最后一层使用softmax
    :return: model
    """
    main_input = Input(shape=(1, 6373), name='main_input')
    x = Dense(64, activation='sigmoid')(main_input)
    x = Dense(64, activation='sigmoid')(x)
    x = Dense(64, activation='sigmoid')(x)

    x = Flatten()(x)

    y_emotion = Dense(8, activation='softmax')(x)

    model = Model(inputs=[main_input], outputs=[y_emotion])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    sgd = optimizers.sgd(lr=0.01)

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.summary()

    return model


def build_model_emotion():
    """
    构建模型, 三个隐藏层, 使用的是sigmoid激活函数
    最后一层使用softmax
    :return: model
    """
    main_input = Input(shape=(1, 6373), name='main_input')

    x = Flatten()(main_input)

    y_emotion = Dense(8, activation='softmax')(x)

    model = Model(inputs=[main_input], outputs=[y_emotion])
    sgd = optimizers.sgd(lr=0.01)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    return model


def build_model_e_g():
    """
    构建模型, 三个隐藏层, 使用的是sigmoid激活函数
    最后一层使用softmax
    :return: model
    """
    main_input = Input(shape=(1, 88), name='main_input')
    x = Dense(64, activation='sigmoid')(main_input)
    x = Dense(64, activation='sigmoid')(x)
    x = Dense(64, activation='sigmoid')(x)

    x = Flatten()(x)

    y_emotion = Dense(8, activation='softmax')(x)
    y_gender = Dense(2, activation='softmax')(x)

    model = Model(inputs=[main_input], outputs=[y_emotion, y_gender])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    sgd = optimizers.sgd(lr=0.01)

    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model.summary()

    return model


def build_model_VAD_M2():
    """
    DNN分层（包含age、gender子属性，但不输出），并行学习出AVD三个属性
    :return:model
    """
    main_input = Input(shape=(1, 6373), name='main_input')

    x = Dense(64, activation='sigmoid')(main_input)

    x = Dropout(rate=0.5)(x)

    x = Flatten()(x)

    v = Dense(1, activation='relu', name='v_out')(x)
    a = Dense(1, activation='relu', name='a_out')(x)
    d = Dense(1, activation='relu', name='d_out')(x)

    model = Model(inputs=[main_input], outputs=[v, a, d])
    sgd = optimizers.sgd(lr=1e-4)
    model.compile(loss={'v_out': 'mse', 'a_out': 'mse', 'd_out': 'mse'},
                  loss_weights={'v_out': 0.2, 'a_out': 0.6, 'd_out': 0.2}, optimizer='rmsprop',
                  metrics=['accuracy'])

    model.summary()

    return model


def build_model_VAD_M1():
    """
    DNN不分层，并行学习VAD三个属性（一次学习）
    :return: model
    """
    main_input = Input(shape=(1, 6373), name='main_input')

    x = Dropout(rate=0.5)(main_input)

    x = Flatten()(x)

    v = Dense(1, activation='relu', name='v_out')(x)
    a = Dense(1, activation='relu', name='a_out')(x)
    d = Dense(1, activation='relu', name='d_out')(x)

    model = Model(inputs=[main_input], outputs=[v, a, d])
    sgd = optimizers.sgd(lr=1e-4)
    model.compile(loss={'v_out': 'mse', 'a_out': 'mse', 'd_out': 'mse'},
                  loss_weights={'v_out': 0.2, 'a_out': 0.6, 'd_out': 0.2}, optimizer='rmsprop',
                  metrics=['accuracy'])

    model.summary()

    return model


def predict_All():
    """
    DNN分层（age、gender），并行学习VAD、age、gender五个属性。
    Age、gender的识别率
    :return:
    """
    main_input = Input(shape=(1, 6373), name='main_input')
    x = Dense(64, activation='sigmoid')(main_input)
    x = Dense(32, activation='sigmoid')(x)
    x = Dense(16, activation='sigmoid')(x)

    x = Flatten()(x)

    y_emotion = Dense(8, activation='softmax', name='emotion_output')(x)

    v_output = concatenate([x, y_emotion], axis=1)
    a_output = concatenate([x, y_emotion], axis=1)
    d_output = concatenate([x, y_emotion], axis=1)
    v = Dense(1, activation='relu', name='v_out')(v_output)
    a = Dense(1, activation='relu', name='a_out')(a_output)
    d = Dense(1, activation='relu', name='d_out')(d_output)

    model = Model(inputs=[main_input], outputs=[y_emotion, v, a, d])

    sgd = optimizers.sgd(1e-4)

    model.compile(
        loss={'emotion_output': 'categorical_crossentropy', 'v_out': concordance_cc, 'a_out': concordance_cc,
              'd_out': concordance_cc},
        optimizer=sgd, metrics=['accuracy'])

    model.summary()

    return model
