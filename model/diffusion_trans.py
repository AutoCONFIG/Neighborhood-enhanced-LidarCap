# -*- coding: utf-8 -*-
# @Author  : jingyi
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath
from functools import partial

# class InputProcess(nn.Module):
#     def __init__(self, data_rep, input_feats, latent_dim):
#         super().__init__()
#         self.data_rep = data_rep
#         self.input_feats = input_feats
#         self.latent_dim = latent_dim
#         self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
#         if self.data_rep == 'rot_vel':
#             self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)
# 
#     def forward(self, x):
#         bs, njoints, nfeats, nframes = x.shape
#         x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)
# 
#         if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
#             x = self.poseEmbedding(x)  # [seqlen, bs, d]
#             return x
#         elif self.data_rep == 'rot_vel':
#             first_pose = x[[0]]  # [1, bs, 150]
#             first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
#             vel = x[1:]  # [seqlen-1, bs, 150]
#             vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
#             return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
#         else:
#             raise ValueError
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_1, x_2, x_3):
        B, N, C = x_1.shape
        q = self.linear_q(x_1).reshape(B, N, self.num_heads,
                                       C // self.num_heads).permute(0, 2, 1, 3)
        k = self.linear_k(x_2).reshape(B, N, self.num_heads,
                                       C // self.num_heads).permute(0, 2, 1, 3)
        v = self.linear_v(x_3).reshape(B, N, self.num_heads,
                                       C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
class CHI_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm3_11 = norm_layer(dim)
        self.norm3_12 = norm_layer(dim)
        self.norm3_13 = norm_layer(dim)

        self.norm3_21 = norm_layer(dim)
        self.norm3_22 = norm_layer(dim)
        self.norm3_23 = norm_layer(dim)

        self.norm3_31 = norm_layer(dim)
        self.norm3_32 = norm_layer(dim)
        self.norm3_33 = norm_layer(dim)

        self.attn_1 = Cross_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
                                      qk_scale=qk_scale, attn_drop=attn_drop,
                                      proj_drop=drop)
        self.attn_2 = Cross_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
                                      qk_scale=qk_scale, attn_drop=attn_drop,
                                      proj_drop=drop)
        self.attn_3 = Cross_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
                                      qk_scale=qk_scale, attn_drop=attn_drop,
                                      proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim * 3)
        self.mlp = Mlp(in_features=dim * 3, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x_1, x_2, x_3):
        x_1 = x_1 + self.drop_path(
            self.attn_1(self.norm3_11(x_2), self.norm3_12(x_3), self.norm3_13(x_1)))
        x_2 = x_2 + self.drop_path(
            self.attn_2(self.norm3_21(x_1), self.norm3_22(x_3), self.norm3_23(x_2)))
        x_3 = x_3 + self.drop_path(
            self.attn_3(self.norm3_31(x_1), self.norm3_32(x_2), self.norm3_33(x_3)))

        x = torch.cat([x_1, x_2, x_3], dim=2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x_1 = x[:, :, :x.shape[2] // 3]
        x_2 = x[:, :, x.shape[2] // 3: x.shape[2] // 3 * 2]
        x_3 = x[:, :, x.shape[2] // 3 * 2: x.shape[2]]

        return x_1, x_2, x_3
        
class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        # input_feats = N joints * 3
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        # self.featureFinal = nn.Linear(self.latent_dim, 1024)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, x_input):
        # output is the features extracted by transformer
        nframes, bs, d = x_input.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(x_input)  # [seqlen, bs, N joints * 3]
            # joints_features = self.featureFinal(x_input)
        elif self.data_rep == 'rot_vel':
            first_pose = x_input[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = x_input[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        # joints_features = joints_features.reshape(nframes, bs, 1024)
        output = output.permute(1, 0, 2, 3)  # [bs, nframes njoints, 3]
        # joints_features = joints_features.permute(1,0,2)
        return output
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=16):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) #(16, 1024)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)#(16,1,1024)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TransEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(TransEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model=d_model
        # pe = pe.unsqueeze(0).transpose(0, 1)  # (16,1,1024)

        # self.register_buffer('te', te)

    def forward(self, x, trans):
        te = torch.zeros(trans.shape[0], trans.shape[1], self.d_model) # (seqlen, bs, d)
        position = torch.norm(trans, dim=-1) #(seqlen, bs)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model)) #(512)
        te[..., 0::2] = torch.sin(position.unsqueeze(-1) * div_term.to(position.device))
        te[..., 1::2] = torch.cos(position.unsqueeze(-1) * div_term.to(position.device))

        x = x + te[:x.shape[0], :].to(x.device)
        return self.dropout(x)
    
    
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)
    
    
class MDM_Trans(nn.Module):
    def __init__(self, cfg):
        super(MDM_Trans, self).__init__()
        self.cfg = cfg
        self.input_feats = self.cfg.MDMTransformer.njoints * self.cfg.MDMTransformer.nfeats
        self.sequence_pos_encoder = PositionalEncoding(self.cfg.MDMTransformer.latent_dim
                                                       , self.cfg.MDMTransformer.dropout,
                                                       self.cfg.MDMTransformer.seqlen)
        self.trans_pos_encoder = TransEncoding(self.cfg.MDMTransformer.latent_dim
                                                       , self.cfg.MDMTransformer.dropout)
        
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.cfg.MDMTransformer.latent_dim,
                                                          nhead=self.cfg.MDMTransformer.num_heads,
                                                          dim_feedforward=self.cfg.MDMTransformer.ff_size,
                                                          dropout=self.cfg.MDMTransformer.dropout,
                                                          activation=self.cfg.MDMTransformer.activation)
        # --------------------------------------------\

        SNTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=self.cfg.MDMTransformer.num_heads,
            dim_feedforward=self.cfg.MDMTransformer.ff_size,
            dropout=self.cfg.MDMTransformer.dropout,
            activation=self.cfg.MDMTransformer.activation)
        

        self.SNTransEncoder = nn.TransformerEncoder(SNTransEncoderLayer,
                                                    num_layers=self.cfg.MDMTransformer.num_layers)

        # -------------------------------------

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.cfg.MDMTransformer.num_layers)
        
        self.embed_timestep = TimestepEmbedder(self.cfg.MDMTransformer.latent_dim, self.sequence_pos_encoder)
        
        self.output_process = OutputProcess(self.cfg.MDMTransformer.data_rep, 
                                            self.input_feats, 
                                            self.cfg.MDMTransformer.latent_dim, 
                                            self.cfg.MDMTransformer.njoints,
                                            self.cfg.MDMTransformer.nfeats)
        
        dpr = [x.item() for x in torch.linspace(0, 0.2, 4)]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(512* 3)
        self.CHI_blocks = nn.ModuleList([
            CHI_Block(
                dim=512, num_heads=8, mlp_hidden_dim=1024,
                qkv_bias=True, qk_scale=None,
                drop=0.1, attn_drop=0., drop_path=dpr[4 - 1],
                norm_layer=norm_layer)
            for i in range(1)])
        # self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

    def forward(self, x_human, x_SN, x_BN, timesteps, trans):
        # x: (bs, seqlen, d)
        x_human = x_human.permute(1, 0, 2)  # (seqlen, bs, d)
        x_SN = x_SN.permute(1, 0, 2)
        x_BN = x_BN.permute(1, 0, 2)
        # trans = trans.permute(1,0,2)

        x_SN_transformer = self.SNTransEncoder(x_SN)

        # x_tmp = torch.cat([x_BN, x_SN_transformer], dim=-1)
        # x = torch.cat([x_human, x_tmp], dim=-1)
        # seq position encoding
        # xseq = self.sequence_pos_encoder(x)  # [seqlen, bs, d]

        # trans position encoding
        # xseq = self.trans_pos_encoder(xseq, trans)

        # output = self.seqTransEncoder(xseq)
        x_human, x_SN, X_BN = self.CHI_blocks[0](x_human, x_SN_transformer, x_BN)
        
        x = torch.cat([x_human, x_SN, X_BN], dim=2)
        x = self.norm(x)
        output = self.seqTransEncoder(x)
    
        output = self.output_process(output)  # [bs, njoints, nfeats, nframes] -to-do-> # (B, T, 24, 3)
        return output
            