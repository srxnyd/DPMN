import math
import torch
from torch import nn
from mamba_ssm import Mamba


_DIAGONAL_INDEX_CACHE = {}


def _get_diagonal_order(H, W, device):
    cache_key = (H, W, device.type, device.index)
    if cache_key in _DIAGONAL_INDEX_CACHE:
        return _DIAGONAL_INDEX_CACHE[cache_key]

    rows = []
    cols = []
    for d in range(H + W - 1):
        if d < W:
            row_start = 0
            col_start = d
        else:
            row_start = d - W + 1
            col_start = W - 1

        row, col = row_start, col_start
        while row < H and col >= 0:
            rows.append(row)
            cols.append(col)
            row += 1
            col -= 1

    row_index = torch.tensor(rows, device=device, dtype=torch.long)
    col_index = torch.tensor(cols, device=device, dtype=torch.long)
    _DIAGONAL_INDEX_CACHE[cache_key] = (row_index, col_index)
    return row_index, col_index


def diagonal_to_sequence(images):
    _, H, W, _ = images.shape
    row_index, col_index = _get_diagonal_order(H, W, images.device)
    return images[:, row_index, col_index, :]


def sequence_to_image(sequences, H, W):
    B, _, C = sequences.shape
    row_index, col_index = _get_diagonal_order(H, W, sequences.device)
    images = torch.zeros((B, H, W, C), device=sequences.device, dtype=sequences.dtype)
    images[:, row_index, col_index, :] = sequences
    return images


class HMamba(nn.Module):
    def __init__(self,channels, token_num=8, use_residual=True, group_num=4):
        super(HMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels/token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba( # This module uses roughly 3 * expand * d_model^2 parameters
                            d_model=self.group_channel_num,  # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=2,  # Block expansion factor
                            )

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self,x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self,x):
        x_pad = self.padding_feature(x)
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        B, H, W, C_pad = x_pad.shape
        x_flat = x_pad.view(B * H * W, self.token_num, self.group_channel_num)
        x_flat = self.mamba(x_flat)
        x_recon = x_flat.view(B, H, W, C_pad)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x_recon)
        if self.use_residual:
            return x + x_proj
        else:
            return x_proj


class VMamba(nn.Module):
    def __init__(self,channels,use_residual=True,group_num=4,use_proj=True):
        super(VMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
                           d_model=channels,  # Model dimension d_model
                           d_state=16,  # SSM state expansion factor
                           d_conv=4,  # Local convolution width
                           expand=2,  # Block expansion factor
                           )
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )

    def forward(self,x):
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B,H,W,C = x_re.shape
        x_flat = x_re.view(1,-1, C)
        x_flat = self.mamba(x_flat)

        x_recon = x_flat.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_recon = self.proj(x_recon)
        if self.use_residual:
            return x_recon + x
        else:
            return x_recon

class diagMamba(nn.Module):
    def __init__(self,channels,use_residual=True,group_num=4,use_proj=True):
        super(diagMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
                           d_model=channels,  # Model dimension d_model
                           d_state=16,  # SSM state expansion factor
                           d_conv=4,  # Local convolution width
                           expand=2,  # Block expansion factor
                           )
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )

    def forward(self,x):
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B,H,W,C = x_re.shape
        x_flat = diagonal_to_sequence(x_re)
        x_flat = self.mamba(x_flat)
        x_recon = sequence_to_image(x_flat, H, W)

        #x_recon = x_flat.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_recon = self.proj(x_recon)
        if self.use_residual:
            return x_recon + x
        else:
            return x_recon


class BothMamba(nn.Module):
    def __init__(self,channels,token_num=4,use_residual=True,group_num=4,use_att=True,use_diag=True):
        super(BothMamba, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        self.use_diag = use_diag
        if self.use_att:
            branch_count = 3 if self.use_diag else 2
            self.weights = nn.Parameter(torch.ones(branch_count) / branch_count)
            self.softmax = nn.Softmax(dim=0)

        self.V_mamba = VMamba(channels,use_residual=use_residual,group_num=group_num)
        # Reuse the vertical mixer on a transposed feature map to model the horizontal direction.
        self.H_mamba = VMamba(channels,use_residual=use_residual,group_num=group_num)
        if self.use_diag:
            self.D_mamba = diagMamba(channels,use_residual=use_residual,group_num=group_num)

    def forward(self,x):
        V_x = self.V_mamba(x)
        x2 = x.permute(0,1,3,2)
        H_x = self.H_mamba(x2)
        H_x = H_x.permute(0,1,3,2)
        branch_outputs = [V_x, H_x]
        if self.use_diag:
            D_x = self.D_mamba(x)
            branch_outputs.append(D_x)
        if self.use_att:
            weights = self.softmax(self.weights)
            fusion_x = sum(branch * weights[idx] for idx, branch in enumerate(branch_outputs))
        else:
            fusion_x = sum(branch_outputs)
        if self.use_residual:
            return fusion_x + x
        else:
            return fusion_x



class MambaHSI(nn.Module):
    def __init__(self,in_channels=128,hidden_dim=64,use_residual=True,mamba_type='cro',token_num=4,group_num=4,use_att=True,use_diag=True):
        super(MambaHSI, self).__init__()
        self.mamba_type = mamba_type

        self.patch_embedding = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=hidden_dim,kernel_size=1,stride=1,padding=0),
                                             nn.GroupNorm(group_num,hidden_dim),
                                             nn.SiLU())

        if mamba_type in ('both', 'cro'):
            self.mamba = nn.Sequential(
                BothMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num,use_att=use_att,use_diag=use_diag),
                BothMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num,use_att=use_att,use_diag=use_diag),
                BothMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num,use_att=use_att,use_diag=use_diag),
            )
        elif mamba_type == 'v':
            self.mamba = VMamba(channels=hidden_dim,use_residual=use_residual,group_num=group_num)
        elif mamba_type == 'h':
            self.mamba = HMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num)
        elif mamba_type == 'diag':
            self.mamba = diagMamba(channels=hidden_dim,use_residual=use_residual,group_num=group_num)
        else:
            raise ValueError(f"Unsupported mamba_type: {mamba_type}")




    def forward(self,x):
        x = self.patch_embedding(x)
        x = self.mamba(x)

        return x



# if __name__=='__main__':
#     batch, length, dim = 2, 512*512, 256
#     x = torch.randn(batch, length, dim).to("cuda")
#     model = Mamba(
#         # This module uses roughly 3 * expand * d_model^2 parameters
#         d_model=dim,  # Model dimension d_model
#         d_state=16,  # SSM state expansion factor
#         d_conv=4,  # Local convolution width
#         expand=2,  # Block expansion factor
#     ).to("cuda")
#     y = model(x)
#     assert y.shape == x.shape
