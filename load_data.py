import numpy as np
import os
from tensorflow.keras.utils import to_categorical

def load_data(dataset: str):
    actions: list[str] = []
    DATA_PATH = f'.\{dataset}\MP_DATA'
    for action in os.listdir(DATA_PATH):
        actions.append(action)
    actions = np.array(actions)
    label_map = {label: num for num, label in enumerate(actions)}
    if 'custom_dataset' in dataset:
        sequences, labels = [], []
        for action in actions:
            for sequence in range(0, 200):
                np_file = str(sequence) + ".npy"
                res = np.load(os.path.join(DATA_PATH, action, np_file))
                sequences.append(res)
                labels.append(label_map[action])
        for action in actions:
            for sequence in range(300, 500):
                np_file = str(sequence) + ".npy"
                res = np.load(os.path.join(DATA_PATH, action, np_file))
                sequences.append(res)
                labels.append(label_map[action])

        X = np.array(sequences)
        y = to_categorical(labels).astype(int)  # thay vi dat so thi dat vi tri 0 1

        sequences_test, labels_test = [], []
        for action in actions:
            for sequence in range(200, 275):
                np_file = str(sequence) + ".npy"
                res = np.load(os.path.join(DATA_PATH, action, np_file))
                sequences_test.append(res)
                labels_test.append(label_map[action])
        for action in actions:
            for sequence in range(500, 525):
                np_file = str(sequence) + ".npy"
                res = np.load(os.path.join(DATA_PATH, action, np_file))
                sequences_test.append(res)
                labels_test.append(label_map[action])
        X_test = np.array(sequences_test)
        y_test = to_categorical(labels_test).astype(int)

        # sequences_valid, labels_valid = [], []
        # for action in actions:
        #     for sequence in range(250, 300):
        #         np_file = str(sequence) + ".npy"
        #         res = np.load(os.path.join(DATA_PATH, action, np_file))
        #         sequences_valid.append(res)
        #         labels_valid.append(label_map[action])
        # for action in actions:
        #     for sequence in range(550, 600):
        #         np_file = str(sequence) + ".npy"
        #         res = np.load(os.path.join(DATA_PATH, action, np_file))
        #         sequences_valid.append(res)
        #         labels_valid.append(label_map[action])
        # X_valid = np.array(sequences_valid)
        # y_valid = to_categorical(labels_valid).astype(int)
        X_valid = None
        y_valid = None
    elif 'wlasl' in dataset:
        sequences, labels = [], []
        for action in actions:
            for sequence in range(0, 100):
                np_file = str(sequence) + ".npy"
                res = np.load(os.path.join(DATA_PATH, action, np_file))
                sequences.append(res)
                labels.append(label_map[action])
        for action in actions:
            for sequence in range(200, 225):
                np_file = str(sequence) + ".npy"
                res = np.load(os.path.join(DATA_PATH, action, np_file))
                sequences.append(res)
                labels.append(label_map[action])

        X = np.array(sequences)
        y = to_categorical(labels).astype(int)  # thay vi dat so thi dat vi tri 0 1

        sequences_test, labels_test = [], []
        for action in actions:
            for sequence in range(150, 200):
                np_file = str(sequence) + ".npy"
                res = np.load(os.path.join(DATA_PATH, action, np_file))
                sequences_test.append(res)
                labels_test.append(label_map[action])
        for action in actions:
            for sequence in range(225, 250):
                np_file = str(sequence) + ".npy"
                res = np.load(os.path.join(DATA_PATH, action, np_file))
                sequences_test.append(res)
                labels_test.append(label_map[action])
        X_test = np.array(sequences_test)
        y_test = to_categorical(labels_test).astype(int)
        X_valid = None
        y_valid = None
    elif 'autsl' in dataset:
        sequences, labels = [], []
        for action in actions:
            for sequence in range(0, 150):
                np_file = str(sequence) + ".npy"
                res = np.load(os.path.join(DATA_PATH, action, np_file))
                sequences.append(res)
                labels.append(label_map[action])
        for action in actions:
            for sequence in range(200, 225):
                np_file = str(sequence) + ".npy"
                res = np.load(os.path.join(DATA_PATH, action, np_file))
                sequences.append(res)
                labels.append(label_map[action])

        X = np.array(sequences)
        y = to_categorical(labels).astype(int)  # thay vi dat so thi dat vi tri 0 1

        sequences_test, labels_test = [], []
        for action in actions:
            for sequence in range(150, 200):
                np_file = str(sequence) + ".npy"
                res = np.load(os.path.join(DATA_PATH, action, np_file))
                sequences_test.append(res)
                labels_test.append(label_map[action])
        for action in actions:
            for sequence in range(225, 250):
                np_file = str(sequence) + ".npy"
                res = np.load(os.path.join(DATA_PATH, action, np_file))
                sequences_test.append(res)
                labels_test.append(label_map[action])
        X_test = np.array(sequences_test)
        y_test = to_categorical(labels_test).astype(int)
        X_valid = None
        y_valid = None
    return actions, X, y, X_test, y_test, X_valid, y_valid
