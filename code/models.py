import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.dim() == 3:
            seq_len = x.size(1)
            pe = self.pe[:seq_len, :].unsqueeze(0)
        else:
            seq_len = x.size(0)
            pe = self.pe[:seq_len, :].unsqueeze(1)
        x = x + pe.to(x.device)
        return x

class MultiFeatureTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len_in=120, seq_len_out=36,
                 d_model=128, nhead=4, num_layers=2):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=0.1,
                activation='gelu'),
            num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation='gelu',
            batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.pos_encoder = PositionalEncoding(d_model)
        self.tgt_embedding = nn.Parameter(torch.randn(seq_len_out, 1, d_model))
        self.output_proj = nn.Linear(d_model, output_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.tgt_embedding)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, src_key_padding_mask=None):
        batch_size = src.size(0)
        src = self.input_proj(src).permute(1, 0, 2)  # [S, N, D]
        src = self.pos_encoder(src)
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        tgt = self.tgt_embedding.repeat(1, batch_size, 1)  # [T_out, N, D]
        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=src_key_padding_mask
        )
        output = self.output_proj(output)
        output = output.permute(1, 0, 2)

        return output
    
class ImprovedBiLSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, seq_len_in=14, seq_len_out=106, num_layers=3, dropout=0.3):
        super().__init__()
        self.seq_len_out = seq_len_out
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.temporal_proj = nn.Linear(seq_len_in, seq_len_out)
        
        self.fc_out = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out.permute(0, 2, 1)
        x = self.temporal_proj(x)
        x = x.permute(0, 2, 1)

        out = self.fc_out(x)
        return out

class GCNSeq2Seq(nn.Module):
    def __init__(self, in_features=4, hidden_dim=64, out_features=2, input_len=14, output_len=106, dropout=0.3):
        super(GCNSeq2Seq, self).__init__()

        self.gcn1 = GCNConv(in_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.temporal_fc = nn.Linear(input_len, output_len)
        self.fc_out = nn.Linear(hidden_dim, out_features)
        self.dropout = dropout

    def forward(self, x, edge_index):
        B, T_in, F_in = x.shape
        x = x.reshape(-1, F_in)  # [B*T_in, F_in]
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)  # [B*T_in, hidden_dim]
        x = x.view(B, T_in, -1)  # [B, 14, hidden_dim]
        x = x.permute(0, 2, 1)
        x = self.temporal_fc(x)
        x = x.permute(0, 2, 1)
        out = self.fc_out(x)
        return out
