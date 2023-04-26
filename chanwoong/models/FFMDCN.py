import tqdm

import numpy as np

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from components import FeaturesLinear, FieldAwareFactorizationMachine, FeaturesEmbedding, CrossNetwork, MultiLayerPerceptron
from components import rmse, RMSELoss


class FFM_DCN(nn.Module):
    """
    A pytorch implementation of Deep & Cross Network.
    Reference:
        R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017,
        baseline DeepCoNN 
    """

    def __init__(self, ff_field_dims: np.ndarray, ff_embed_dim: int, dcn_embed_dim: int, num_layers: int, mlp_dims: tuple, dropout: float):
        super().__init__()
        self.ff_linear = FeaturesLinear(ff_field_dims)
        self.ffm = FieldAwareFactorizationMachine(ff_field_dims, ff_embed_dim)
        self.dcn_embedding = FeaturesEmbedding(ff_field_dims[:2], dcn_embed_dim)
        self.embed_output_dim = len(ff_field_dims[:2]) * dcn_embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.cd_linear = nn.Linear(mlp_dims[0], 1, bias=False)
        self.linear = nn.Linear(2,1,bias=False)

    def forward(self, ffx: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        dcnx = ffx[:,:2]
        ffm_term = torch.sum(torch.sum(self.ffm(ffx), dim=1), dim=1, keepdim=True)
        ffx = self.ff_linear(ffx) + ffm_term
        embed_x = self.dcn_embedding(dcnx).view(-1, self.embed_output_dim)
        x_l1 = self.cn(embed_x)
        x_out = self.mlp(x_l1)
        p = self.cd_linear(x_out)
        p = self.linear(torch.cat([ffx,p], dim=1))
        return p.squeeze(1)

    
class FFDCNModel:

    def __init__(self, args, dataffm):
        super().__init__()

        self.criterion = RMSELoss()

        self.ff_train_dataloader = dataffm['train_dataloader']
        self.ff_valid_dataloader = dataffm['valid_dataloader']
        self.ff_field_dims = dataffm['field_dims']

        self.ff_embed_dim = args.FFM_EMBED_DIM
        self.dcn_embed_dim = args.DCN_EMBED_DIM
        self.epochs = args.epochs
        self.learning_rate = args.lr
        self.weight_decay = args.weight_decay
        self.log_interval = 100
        
        self.args = args
        self.idx2user = dataffm['idx2user']
        self.idx2isbn = dataffm['idx2isbn']

        self.device = args.device

        self.mlp_dims = args.DCN_MLP_DIMS
        self.dropout = args.DCN_DROPOUT
        self.num_layers = args.DCN_NUM_LAYERS

        self.model = FFM_DCN(self.ff_field_dims, self.ff_embed_dim, self.dcn_embed_dim, num_layers=self.num_layers, mlp_dims=self.mlp_dims, dropout=self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.ff_train_dataloader, smoothing=0, mininterval=1.0)
            for i, (ff_fields, target) in enumerate(tk0):
                ff_fields, target = ff_fields.to(self.device), target.to(self.device)
                y = self.model(ff_fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)


    def predict_train(self,save=False):
        self.model.eval()
        targets, predicts = list(), list()
        users, isbns = np.array([]),np.array([])
        with torch.no_grad():
            for ff_fields, target in tqdm.tqdm(self.ff_valid_dataloader, smoothing=0, mininterval=1.0):
                ff_fields, target = ff_fields.to(self.device), target.to(self.device)
                y = self.model(ff_fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
                if save:
                    users = np.concatenate((users,ff_fields[:,0].tolist()))
                    isbns = np.concatenate((isbns,ff_fields[:,1].tolist()))
            if save:
                print(f'--------------- Saving Valid ---------------')
                df_valid = pd.DataFrame({
                    'user_id':users,
                    'isbn':isbns,
                    'target':targets,
                    'rating':predicts})
                df_valid['user_id'] = df_valid['user_id'].map(self.idx2user)
                df_valid['isbn'] = df_valid['isbn'].map(self.idx2isbn)
        return rmse(targets, predicts)


    def predict(self, ff_dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for ff_fields in tqdm.tqdm(ff_dataloader, smoothing=0, mininterval=1.0):
                ff_fields = ff_fields[0].to(self.device)
                y = self.model(ff_fields)
                predicts.extend(y.tolist())
        return predicts