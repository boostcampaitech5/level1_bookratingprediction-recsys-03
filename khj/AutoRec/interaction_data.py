import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset


def list_to_arr(lst, idx, arr: np.ndarray) -> np.ndarray:
    user, items, ratings = lst[idx]
    for item, rating in zip(items, ratings):
        arr[item] += rating
  
    return arr


def interaction_data_load(args):
    """
    Parameters
    ----------
    Args:
        data_path : str
            데이터 경로
    ----------
    """

    ######################## DATA LOAD
    train = pd.read_csv(args.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.data_path + 'test_ratings.csv')
    ratings = pd.concat([train, test])
    sub = pd.read_csv(args.data_path + 'sample_submission.csv')
    
    user2idx = {v:k for k, v in enumerate(ratings['user_id'].unique())}
    isbn2idx = {v:k for k, v in enumerate(ratings['isbn'].unique())}
    
    train['isbn'] = train['isbn'].map(isbn2idx)
    train['user_id'] = train['user_id'].map(user2idx)
    
    train_result = train.groupby('user_id').apply(lambda x: [x['user_id'].tolist()[0], x['isbn'].tolist(), x['rating'].tolist()]).values.tolist()
    
    test['isbn'] = test['isbn'].map(isbn2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    
    #test_result = test.groupby('user_id').apply(lambda x: [x['user_id'].tolist(), x['isbn'].tolist(), x['rating'].tolist()]).values.tolist()
    
    
    data = {
            'train':train_result,
            'test':test.drop(['rating'], axis=1),
            'sub':sub,
            'input_dim':len(isbn2idx),
            'user2idx':user2idx,
            'isbn2idx':isbn2idx
            }

    return data


def interaction_data_split(args, data):
    """
    Parameters
    ----------
    Args:
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            랜덤 seed 값
    ----------
    """

    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'],
                                                        data['train'],
                                                        test_size=0.33, random_state=42,
                                                        shuffle=True
                                                        )
    
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data


class TrainDataset(Dataset):
    def __init__(self, X, y, input_dim):
        self.X = X
        self.y = y
        self.input_dim = input_dim
        
    def __getitem__(self, index):
        arr = np.zeros(self.input_dim)
        return list_to_arr(self.X, index, arr).astype('float32'), list_to_arr(self.y, index, arr).astype('float32')
        
    def __len__(self):
        return len(self.X)


class TestDataset(Dataset):
    def __init__(self, test, train, input_dim):
        self.test = test
        self.train = train
        self.input_dim = input_dim
        
    def __getitem__(self, index):
        user_id, isbn = self.test.iloc[index]
        arr = np.zeros(self.input_dim)
        try:
            return list_to_arr(self.train, train.index(user_id), arr).astype('float32'), isbn 
        except:
            return arr.astype('float32'), isbn
        
    def __len__(self):
        return len(self.test)


def interaction_data_loader(args, data):
    """
    Parameters
    ----------
    Args:
        batch_size : int
            데이터 batch에 사용할 데이터 사이즈
        data_shuffle : bool
            data shuffle 여부
    ----------
    """

    #train_dataset = TensorDataset(torch.LongTensor(data['X_train']), torch.LongTensor(data['y_train']))
    #valid_dataset = TensorDataset(torch.LongTensor(data['X_valid']), torch.LongTensor(data['y_valid']))
    #test_dataset = TensorDataset(torch.LongTensor(data['test']))
    
    train_dataset = TrainDataset(data['X_train'], data['y_train'], data['input_dim'])
    valid_dataset = TrainDataset(data['X_valid'], data['y_valid'], data['input_dim'])
    test_dataset = TestDataset(data['test'], data['train'], data['input_dim'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data