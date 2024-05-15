import json
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, MaxPooling2D , Flatten, Conv2D, TimeDistributed
from load_data import load_data

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
dataset = 'custom_dataset'
print(dataset)
actions, X, y, X_test, y_test, X_valid, y_valid = load_data(dataset)
BATCH_SIZE = 64 if dataset == "wlasl_300" else 128
NUM_EPOCHS = 100 if dataset == "autsl" else 40
filters = [32, 64, 128]
kernel_sizes = [4, 5, 6]
filter_1 = filters[1]
filter_2 = filters[1]  
results = {}
for filter_3 in filters:
    for filter_4 in filters:
        for kernel_size_1 in kernel_sizes:
            for kernel_size_2 in kernel_sizes:
                for kernel_size_3 in kernel_sizes:
                    for kernel_size_4 in kernel_sizes:
                        print(f"{filter_1}_{filter_2}_{filter_3}_{filter_4}_{kernel_size_1}_{kernel_size_2}_{kernel_size_3}_{kernel_size_4}")
                        model = Sequential()
                        model.add(TimeDistributed(Conv1D(filters=filter_1, kernel_size=kernel_size_1, activation="relu"), input_shape=(7, 46, 2)))
                        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
                        model.add(TimeDistributed(Conv1D(filters=filter_2, kernel_size=kernel_size_2, activation="relu")))
                        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
                        model.add(Conv2D(filters=filter_3, kernel_size=(kernel_size_3, kernel_size_3), padding='same', activation='relu'))
                        model.add(MaxPooling2D(pool_size=2, strides=1, padding='same'))
                        model.add(Conv2D(filters=filter_4, kernel_size=(kernel_size_4, kernel_size_4), padding='same', activation='relu'))
                        model.add(MaxPooling2D(pool_size=2, strides=1, padding='same'))
                        model.add(Flatten())
                        model.add(Dense(100, activation='relu'))
                        model.add(Dense(100, activation='relu'))
                        model.add(Dense(100, activation='relu'))
                        model.add(Dense(len(actions), activation='softmax')) 
                        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
                        # Train AI model
                        history = model.fit(X, y, epochs=NUM_EPOCHS, validation_data=(X_test, y_test), batch_size=BATCH_SIZE)
                        model.summary()
                        result = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
                        results[f"2TDCNN2CNN3D_{filter_1}_{filter_2}_{filter_3}_{filter_4}_{kernel_size_1}_{kernel_size_2}_{kernel_size_3}_{kernel_size_4}_loss"] = result[0]
                        results[f"2TDCNN2CNN3D_{filter_1}_{filter_2}_{filter_3}_{filter_4}_{kernel_size_1}_{kernel_size_2}_{kernel_size_3}_{kernel_size_4}_acc"] = result[1]

with open(f"2TDCNN2CNN3D_{filter_1}_{filter_2}.json", "w") as outfile:
    json.dump(results, outfile)

