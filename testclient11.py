import argparse
import os
from pathlib import Path

import tensorflow as tf

import flwr as fl
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

model = load_model('dl_model.h5')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        self.model.set_weights(parameters)

        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        self.model.set_weights(parameters)

        steps: int = config["val_steps"]

        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def load_partition(num):
    csv_file_path = f'experiments/master_experiment_{num}.csv'
    data = pd.read_csv(csv_file_path)

    X = data.iloc[:, 1:4].values
    y = data.iloc[:, -1].values
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(normalized_data, y, test_size=0.2, random_state=42)

    return (
               x_train,
               y_train,
           ), (
               x_test,
               y_test,
           )


#
# model = Sequential()
# model.add(Dense(256, input_dim=3, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification, adjust as needed

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# (x_train1, y_train1), (x_test1, y_test1) = load_partition(1)
#
# client1 = CifarClient(model, x_train1, y_train1, x_test1, y_test1)
#
# fl.client.start_numpy_client(
#     server_address="127.0.0.1:8080",
#     client=client1,
#
# )
#
# (x_train2, y_train2), (x_test2, y_test2) = load_partition(2)
#
# client2 = CifarClient(model, x_train2, y_train2, x_test2, y_test2)
#
# fl.client.start_numpy_client(
#     server_address="127.0.0.1:8080",
#     client=client2,
#
# )
#
# (x_train3, y_train3), (x_test3, y_test3) = load_partition(3)
# client3 = CifarClient(model, x_train3, y_train3, x_test3, y_test3)
#
# fl.client.start_numpy_client(
#     server_address="127.0.0.1:8080",
#     client=client3,
#
# )
#
# (x_train4, y_train4), (x_test4, y_test4) = load_partition(4)
# client4 = CifarClient(model, x_train4, y_train4, x_test4, y_test4)
#
# fl.client.start_numpy_client(
#     server_address="127.0.0.1:8080",
#     client=client4,
#
# )
#
# (x_train5, y_train5), (x_test5, y_test5) = load_partition(5)
#
# client5 = CifarClient(model, x_train5, y_train5, x_test5, y_test5)
#
# fl.client.start_numpy_client(
#     server_address="127.0.0.1:8080",
#     client=client5,
#
# )
#
# (x_train6, y_train6), (x_test6, y_test6) = load_partition(6)
# client6 = CifarClient(model, x_train6, y_train6, x_test6, y_test6)
#
# fl.client.start_numpy_client(
#     server_address="127.0.0.1:8080",
#     client=client6,
#
# )
#
# (x_train7, y_train7), (x_test7, y_test7) = load_partition(7)
# client7 = CifarClient(model, x_train7, y_train7, x_test7, y_test7)
#
# fl.client.start_numpy_client(
#     server_address="127.0.0.1:8080",
#     client=client7,
#
# )
#
# (x_train8, y_train8), (x_test8, y_test8) = load_partition(8)
# client8 = CifarClient(model, x_train8, y_train8, x_test8, y_test8)
#
# fl.client.start_numpy_client(
#     server_address="127.0.0.1:8080",
#     client=client8,
#
# )
#
# (x_train9, y_train9), (x_test9, y_test9) = load_partition(9)
# client9 = CifarClient(model, x_train9, y_train9, x_test9, y_test9)
#
# fl.client.start_numpy_client(
#     server_address="127.0.0.1:8080",
#     client=client9,
#
# )
#
# (x_train10, y_train10), (x_test10, y_test10) = load_partition(10)
# client10 = CifarClient(model, x_train10, y_train10, x_test10, y_test10)
#
# fl.client.start_numpy_client(
#     server_address="127.0.0.1:8080",
#     client=client10,
#
# )

(x_train11, y_train11), (x_test11, y_test11) = load_partition(11)
client11 = CifarClient(model, x_train11, y_train11, x_test11, y_test11)

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=client11,

)

# (x_train12, y_train12), (x_test12, y_test12) = load_partition(12)
# client12 = CifarClient(model, x_train12, y_train12, x_test12, y_test12)
#
# fl.client.start_numpy_client(
#     server_address="127.0.0.1:8080",
#     client=client12,
#
# )
