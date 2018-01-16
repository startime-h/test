from keras import Input
from keras import optimizers
from keras.layers import Dense, Flatten
from keras.models import Model


def build_model_emotion1():
    """
    构建模型, 三个隐藏层, 使用的是sigmoid激活函数
    最后一层使用softmax
    :return: model
    """
    main_input = Input(shape=(1, 88), name='main_input')
    x = Dense(64, activation='sigmoid')(main_input)
    x = Dense(64, activation='sigmoid')(x)

    x = Flatten()(x)

    y_emotion = Dense(8, activation='softmax')(x)

    model = Model(inputs=[main_input], outputs=[y_emotion])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    sgd = optimizers.sgd(lr=0.01)

    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model.summary()

    return model


def build_model_emotion():
    """
    构建模型, 三个隐藏层, 使用的是sigmoid激活函数
    最后一层使用softmax
    :return: model
    """
    main_input = Input(shape=(1, 88), name='main_input')

    x = Flatten()(main_input)

    y_emotion = Dense(8, activation='softmax')(x)

    model = Model(inputs=[main_input], outputs=[y_emotion])
    sgd = optimizers.sgd(lr=0.01)
    model.compile(optimizer=sgd,
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


def build_model_v_a_d():
    main_input = Input(shape=(1, 88), name='main_input')
    x = Dense(64, activation='sigmoid')(main_input)
    x = Dense(32, activation='sigmoid')(main_input)

    x = Flatten()(x)

    v = Dense(1, activation='softplus')(x)
    # a = Dense(1, activation='softplus')(x)
    # d = Dense(1, activation='softplus')(x)

    model = Model(inputs=[main_input], outputs=v)
    sgd = optimizers.sgd(lr=0.01)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model.summary()
    return model


def my_model():
    main_input = Input(shape=(1, 88), name='main_input')
    x = Dense(64, activation='tanh')(main_input)
    x = Dense(32, activation='tanh')(x)
    x = Dense(16, activation='tanh')(x)

    x = Flatten()(x)

    y_emotion = Dense(4, activation='softmax')(x)
    y_gender = Dense(2, activation='sigmoid')(x)
    v = Dense(1, activation='tanh')(x)
    a = Dense(1, activation='tanh')(x)
    d = Dense(1, activation='tanh')(x)

    model = Model(inputs=[main_input], outputs=[y_emotion, y_gender, v, a, d])
    sgd = optimizers.sgd(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model
