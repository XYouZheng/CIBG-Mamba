import torch
import torch.nn as nn
from layers.Mamba_RNN import InteractiveBidirectionalMambaGRU
from layers.Embed import Channel_Independencies_Embedding
from einops import rearrange

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.d_model=configs.d_model
        self.patch_size=configs.patch_size
        self.stride=configs.stride
        self.device = configs.devices
        self.enc_embedding = Channel_Independencies_Embedding(configs.seq_len, configs.d_model, configs.patch_size, configs.stride, configs.embed, configs.freq, configs.dropout)
        self.encoder =InteractiveBidirectionalMambaGRU(
                    d_model=configs.d_model,  
                    d_state=configs.d_state,
                    dropout=configs.dropout
                )      
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev 
        B, L, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out