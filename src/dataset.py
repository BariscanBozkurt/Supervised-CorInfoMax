import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torch.nn.functional as F

def get_boston_housing_dataset_pytorch(batch_size = 10, test_size_percent = 0.2, seed = 10):
    bos = load_boston()
    df = pd.DataFrame(bos.data)
    df.columns = bos.feature_names
    df['Price'] = bos.target
    data = df[df.columns[:-1]]

    data['Price'] = df.Price
    X = data.drop('Price', axis=1).to_numpy()
    Y = data['Price'].to_numpy()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size_percent, random_state = seed)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.astype(np.float32))
    X_test = scaler.transform(X_test.astype(np.float32))

    X_train = torch.tensor(X_train, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    max_Y_train = Y_train.max()
    Y_train = torch.tensor(Y_train, dtype=torch.float).view(-1, 1) / max_Y_train
    Y_test = torch.tensor(Y_test, dtype=torch.float).view(-1, 1) / max_Y_train
    trn_datasets = torch.utils.data.TensorDataset(X_train, Y_train)
    tst_datasets = torch.utils.data.TensorDataset(X_test, Y_test)
    train_loader = torch.utils.data.DataLoader(trn_datasets, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(tst_datasets, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, max_Y_train