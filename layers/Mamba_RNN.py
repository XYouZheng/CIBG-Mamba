import torch.nn as nn
from mamba_ssm import Mamba,BidirectionalMamba
import torch
from einops import rearrange

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                x, attn = attn_layer(x)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class InteractiveBidirectionalMambaGRU(nn.Module):
    def __init__(self, d_model, d_state, dropout=0.1, reduction=16):
        super(InteractiveBidirectionalMambaGRU, self).__init__()
        self.d_model = d_model
        
        # Mamba layers
        self.mamba_z = Mamba(d_model, d_state, d_conv=4, expand=1)
        self.mamba_h = Mamba(d_model, d_state, d_conv=4, expand=1)
        self.mamba_r = Mamba(d_model, d_state, d_conv=4, expand=1)
        # Interactive layers
        self.interactive_forward = nn.Linear(2*d_model, d_model)
        self.interactive_backward = nn.Linear(2*d_model, d_model)
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x: (batch_size, num_patches, n, d_model)
        batch_size, num_patches, n, d_model = x.size()
        
        # Reshape input to (batch_size * n, num_patches, d_model)
        x_reshaped = x.transpose(1, 2).reshape(batch_size * n, num_patches, d_model)
        
        # Compute Mamba outputs for all patches
        z = self.sigmoid(self.mamba_z(x_reshaped))
        r = self.sigmoid(self.mamba_r(x_reshaped))
        h_candidate = self.tanh(self.mamba_h(x_reshaped))
        
        # Initialize hidden states
        h_forward = torch.zeros(batch_size * n, num_patches, d_model, device=x.device)
        h_backward = torch.zeros(batch_size * n, num_patches, d_model, device=x.device)
        
        # Bidirectional processing with interaction
        for p in range(num_patches):
            # Forward processing
            if p > 0:
                h_interactive_forward = self.interactive_forward(torch.cat([h_forward[:, p-1], h_backward[:, p]], dim=-1))
            else:
                h_interactive_forward = torch.zeros_like(h_forward[:, 0])
            
            h_reset_forward = r[:, p] * h_interactive_forward
            h_new_forward = self.tanh(h_candidate[:, p] + h_reset_forward)
            h_forward[:, p] = (1 - z[:, p]) * h_interactive_forward + z[:, p] * h_new_forward

            # Backward processing
            back_p = num_patches - 1 - p
            if back_p < num_patches - 1:
                h_interactive_backward = self.interactive_backward(torch.cat([h_forward[:, back_p], h_backward[:, back_p+1]], dim=-1))
            else:
                h_interactive_backward = torch.zeros_like(h_backward[:, back_p])
            
            h_reset_backward = r[:, back_p] * h_interactive_backward
            h_new_backward = self.tanh(h_candidate[:, back_p] + h_reset_backward)
            h_backward[:, back_p] = (1 - z[:, back_p]) * h_interactive_backward + z[:, back_p] * h_new_backward
        final_forward = h_forward[:, -1, :]
        final_backward = h_backward[:, 0, :]
        
        combined_output = final_forward+ final_backward
        # print(combined_output.shape)
        # # Reshape combined output back to (batch_size, n, d_model)
        combined_output = combined_output.view(batch_size, n, d_model)
        
        # Apply normalization and dropout
        final_output = self.norm(combined_output)
        final_output = self.dropout(final_output)
        
        return final_output, None