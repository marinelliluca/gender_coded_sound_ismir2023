import torch
import torch.nn as nn

class Backend(nn.Module):
    '''
    Inspired by https://github.com/minzwon/sota-music-tagging-models/
    '''
    def __init__(self,main_dict, 
                 bert_config = None):
        super(Backend, self).__init__()

        backend_dict = main_dict["backend_dict"]
        self.frontend_out_channels = main_dict["frontend_dict"]["list_out_channels"][-1]

        self.seq2seq = None
        # seq2seq for position encoding
        if backend_dict["recurrent_units"] is not None:
            # input and output = (seq_len, batch, frontend_out_channels)
            # (seq_len, batch, 2 * frontend_out_channels) if bidirectional = True
            self.seq2seq = nn.GRU(self.frontend_out_channels, 
                                  self.frontend_out_channels, 
                                  backend_dict["recurrent_units"], 
                                  bidirectional = backend_dict["bidirectional"])

        self.m = 2 if backend_dict["bidirectional"] else 1
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.m*self.frontend_out_channels,8), 
                                                         num_layers=2)
        
        self.single_cls = self.get_cls()
        
        self.dense1 = nn.Linear(self.m*self.frontend_out_channels, self.frontend_out_channels)
        
        # Dense
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(self.frontend_out_channels, backend_dict["n_class"])
        
    def get_cls(self,):
        torch.manual_seed(42) # for reproducibility
        single_cls = torch.rand((1,self.m*self.frontend_out_channels))
        return single_cls

    def append_cls(self, x):
        # insert always the same token as a reference for classification
        vec_cls = self.single_cls.repeat(1,x.shape[1],1) # batch_size = x.shape[1]
        vec_cls = vec_cls.to(x.device)
        return torch.cat([vec_cls, x], dim=0)
        
    def forward(self, seq):

        # frontend output shape = (batch, features, sequence)
        # input to multihead attention and recurrent unit (sequence, batch, features)
        seq = seq.permute(2,0,1)
        
        if self.seq2seq is not None:
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.seq2seq.flatten_parameters() 
            seq,_ = self.seq2seq(seq)
        
        # Self-attention
        seq = self.append_cls(seq)        
        seq = self.transformer_encoder(seq)
        
        # Pool by taking the first (CLS) token
        x = seq[0,:,:]
        x = self.dense1(x)
        x = nn.ReLU()(x)
        
        # Dense
        x = self.dropout(x.squeeze())
        x = self.dense2(x)
        x = nn.Sigmoid()(x)

        return x