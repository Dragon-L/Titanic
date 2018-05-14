import keras as k
from keras.layers import Dense, Dropout


def training_parameter(x_train, y_train, x_dev, y_dev, x_test):
    model = k.models.Sequential()

    model.add(Dense(units=7, activation='relu', input_dim=x_train.shape[1]))
    # model.add(Dropout(rate=0.2))
    model.add(Dense(units=5, activation='relu'))
    model.add(Dense(units=5, activation='relu'))
    model.add(Dense(units=3, activation='relu'))
    model.add(Dense(units=3, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=2000, verbose=2)
    print(history.history['loss'][-1], history.history['acc'][-1])
    loss_and_metrics = model.evaluate(x_dev, y_dev)
    print(loss_and_metrics)

    predict_outcome = model.predict(x_test)
    return predict_outcome
