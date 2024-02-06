from typing import Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import flwr as fl
from keras.models import load_model


model = load_model('dl_model.h5')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    csv_file_path = 'experiment_1_randomised_data.csv'
    data = pd.read_csv(csv_file_path)

    X = data.iloc[:, 1:4].values
    y = data.iloc[:, -1].values
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(X)
    x_train, x_val, y_train, y_val = train_test_split(normalized_data, y, test_size=0.2, random_state=42)
    # model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_val, y_val))
    # loss, accuracy = model.evaluate(x_val, y_val)
    # print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        print(accuracy)
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(server_round: int):
    config = {
        "batch_size": 32,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.3,
    fraction_evaluate=0.2,
    min_fit_clients=12,
    min_evaluate_clients=2,
    min_available_clients=12,
    evaluate_fn=get_evaluate_fn(model),
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
    initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=4),
    strategy=strategy,


)
