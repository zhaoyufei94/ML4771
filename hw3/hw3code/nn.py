import scipy.io as sio
mat = sio.loadmat("MSData.mat")
testx, trainx, trainy = mat["testx"], mat["trainx"], mat["trainy"]

train_data = trainx[:300000]
train_targets = trainy[:300000]
test_data = trainx[300000:]
test_targets = trainy[300000:]

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

from keras import models, layers
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mae', metrics=['mae'])
    return model


"""
K-fold Validation


import numpy as np

k = 5
num_val = len(train_data) // k
num_epochs = 1000
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val: (i + 1) * num_val]
    val_targets = train_targets[i * num_val: (i + 1) * num_val]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val],
        train_data[(i + 1) * num_val:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val],
        train_targets[(i + 1) * num_val:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
        validation_data=(val_data, val_targets), epochs = num_epochs, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_history = smooth_curve(average_mae_history[10:])

import matplotlib.pyplot as plt
plt.plot(range(1, len(smooth_history) + 1), smooth_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
"""

history = model.fit(train_data, train_targets, epochs=1000, verbose=0)
test_mae_score, test_mae = model.evaluate(test_data, test_targets)
print(test_mae_score)
# 6.56

testx -= mean
testx /= std
pre=model.predict(testx)

with open("prediction.csv", "w") as f:
    f.write("dataid,prediction\n")
    for i in range(51630):
        f.write("{},{}\n".format(i+1, int(pre[i, 0])))