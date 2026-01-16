import pandas as pd

def get_federated_datasets_random(num_clients=5):
    datasets = []
    for i in range(num_clients):
        x = pd.read_csv(f'../federated_data_random/client_random_{i+1}.csv')
        datasets.append(x)
    return datasets

def get_federated_datasets_sensitive(num_clients=5):
    datasets = []
    for i in range(num_clients):
        x = pd.read_csv(f'../federated_data_sensitive/client_skewed_{i+1}.csv')
        datasets.append(x)
    return datasets

def get_centralized_dataset():
    data = pd.read_csv('../data/centralized_dataset.csv')
    return data