import numpy as np
import torch
import torch.nn as nn

class AutoRec(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.input_dim = data['input_dim']
        self.embed_dim = args.embed_dim
        
        self.encoder = nn.Linear(self.input_dim, self.embed_dim)
        self.decoder = nn.Linear(self.embed_dim, self.input_dim)
        
        torch.nn.init.xavier_uniform_(self.encoder.weight.data)
        torch.nn.init.xavier_uniform_(self.decoder.weight.data)


    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        h = nn.functional.relu(h)
        
        output = self.decoder(h)
        output = nn.functional.relu(output)
        
        return output
        