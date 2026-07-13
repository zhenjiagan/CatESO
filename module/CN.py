import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
from .Transformer import  TransformerLayer, CrossTransformerLayer, MultiHeadAttention
from timm.models.layers import trunc_normal_
import math

class MCDC(nn.Module):
    def __init__(self, channels, kernel_list=None, K=4, use_bn=True, activation='relu'):
        super().__init__()
        self.channels = channels
        self.K = K
        
        if kernel_list is None:
            kernel_list = [3]
        self.kernel_list = kernel_list
        self.num_kernels = len(kernel_list)
        
        self.weight_generators = nn.ModuleList()
        self.paddings = []
        
        for k_size in kernel_list:
            generator = nn.Linear(channels, K * channels * k_size)
            
            nn.init.kaiming_normal_(generator.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(generator.bias, 0)
            
            self.weight_generators.append(generator)
            self.paddings.append(k_size // 2)
        
        self.bias = nn.Parameter(torch.zeros(channels))

        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(channels)
        
        self.activation = None
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()

    def forward(self, x, x_guide, x_mask=None, guide_mask=None):
        B, L, C = x.shape
        
        if guide_mask is not None:
             guide_mask_expanded = guide_mask.unsqueeze(2)
             guide_masked = x_guide.masked_fill(guide_mask_expanded, 0)
             valid_lengths = (~guide_mask).float().sum(dim=1, keepdim=True).clamp(min=1)
             global_feat = guide_masked.sum(dim=1) / valid_lengths # [B, C]
        else:
             global_feat = x_guide.mean(dim=1) # [B, C]
        
        out_all = 0.0
        x_in = x.transpose(1, 2).reshape(1, B * C, L)
        
        for idx, k_size in enumerate(self.kernel_list):
            generated_params = self.weight_generators[idx](global_feat)
            generated_params = generated_params.view(B, self.K, C, k_size)
            agg_weight = generated_params.sum(dim=1) 
            w_in = agg_weight.view(B * C, 1, k_size)
            
            out = F.conv1d(x_in, w_in, bias=None, padding=self.paddings[idx], groups=B * C)
            out = out.view(B, C, L) # [B, C, L]
            
            out_all = out_all + out
        
        out_all = out_all + self.bias.view(1, -1, 1)

        divider = self.num_kernels * self.K
        out_all = out_all / (divider ** 0.5)

        if self.use_bn:
            out_all = self.bn(out_all)
        
        out_all = out_all.transpose(1, 2)  # [B, L, C]
        
        if self.activation is not None:
            out_all = self.activation(out_all)
        
        if x_mask is not None:
            out_all = out_all.masked_fill(x_mask.unsqueeze(2), 0)
            
        return out_all

class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out

class CrossAttention(nn.Module):
    def __init__(self, dim_model=128, num_head=4, attn_pdrop=.1, resid_pdrop=.1):
        super(CrossAttention, self).__init__()
        self.dim_model = dim_model
        self.dim_head = dim_model // num_head
        self.num_head = num_head

        self.que_proj_pro = nn.Linear(self.dim_model, self.dim_model, bias=False)  # query projection
        self.key_proj_pro = nn.Linear(self.dim_model, self.dim_model, bias=False)  # key projection
        self.val_proj_pro = nn.Linear(self.dim_model, self.dim_model, bias=False)  # value projection

        self.que_proj_drg = nn.Linear(self.dim_model, self.dim_model, bias=False)  # query projection
        self.key_proj_drg = nn.Linear(self.dim_model, self.dim_model, bias=False)  # key projection
        self.val_proj_drg = nn.Linear(self.dim_model, self.dim_model, bias=False)  #  value projection

        self.out_proj_pro = nn.Linear(self.dim_model, self.dim_model)  # output projection
        self.out_proj_drg = nn.Linear(self.dim_model, self.dim_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(self.dim_model)
        self.LN2 = nn.LayerNorm(self.dim_model)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x_p, x_d, p_mask=None, d_mask=None):
        x_pro = x_p
        x_drg = x_d
        b_s = x_pro.shape[0]

        # Layer normalization
        x_pro = self.LN1(x_pro)
        q_pro = self.que_proj_pro(x_pro).contiguous().view(b_s, -1, self.num_head, self.dim_head).permute(0, 2, 1, 3)  # (b_s, h, seq_len, d_k)
        k_pro = self.key_proj_pro(x_pro).contiguous().view(b_s, -1, self.num_head, self.dim_head).permute(0, 2, 3, 1)  # (b_s, h, d_k, seq_len) K^T
        v_pro = self.val_proj_pro(x_pro).contiguous().view(b_s, -1, self.num_head, self.dim_head).permute(0, 2, 1, 3)  # (b_s, h, seq_len, d_v)

        x_drg = self.LN2(x_drg)
        q_drg = self.que_proj_drg(x_drg).contiguous().view(b_s, -1, self.num_head, self.dim_head).permute(0, 2, 1, 3)  # (b_s, h, seq_len, d_k)
        k_drg = self.key_proj_drg(x_drg).contiguous().view(b_s, -1, self.num_head, self.dim_head).permute(0, 2, 3, 1)  # (b_s, h, d_k, seq_len) K^T
        v_drg = self.val_proj_drg(x_drg).contiguous().view(b_s, -1, self.num_head, self.dim_head).permute(0, 2, 1, 3)  # (b_s, h, seq_len, d_v)

        att_pro = torch.matmul(q_pro, k_drg) / math.sqrt(self.dim_head)  # (b_s, h, seq_len, seq_len)
        att_drg = torch.matmul(q_drg, k_pro) / math.sqrt(self.dim_head)  # (b_s, h, seq_len, seq_len)

        if d_mask is not None:
            att_pro = att_pro.masked_fill(d_mask.unsqueeze(1).unsqueeze(2), -1e9)  # mask drug keys
        if p_mask is not None:
            att_drg = att_drg.masked_fill(p_mask.unsqueeze(1).unsqueeze(2), -1e9)  # mask protein keys

        att_pro = torch.softmax(att_pro, -1)
        att_pro = self.attn_drop(att_pro)
        att_drg = torch.softmax(att_drg, -1)
        att_drg = self.attn_drop(att_drg)

        out_pro = torch.matmul(att_pro, v_drg).permute(0, 2, 1, 3).contiguous().view(b_s, -1, self.num_head * self.dim_head)  # (b_s, seq_len, h*d_v)
        out_pro = self.resid_drop(self.out_proj_pro(out_pro))  # (b_s, seq_len, d_model)
        out_drg = torch.matmul(att_drg, v_pro).permute(0, 2, 1, 3).contiguous().view(b_s, -1, self.num_head * self.dim_head)  # (b_s, seq_len, h*d_v)
        out_drg = self.resid_drop(self.out_proj_drg(out_drg))  # (b_s, seq_len, d_model)

        return out_pro, out_drg

class CrossTransformerBlock(nn.Module):
    def __init__(self, dim_model=128, num_head=4, attn_pdrop=.1, resid_pdrop=.1, loops_num=None, block_exp=4):
        super(CrossTransformerBlock, self).__init__()
        print(f"loops_num: {loops_num}")
        self.loops = loops_num if loops_num is not None else 1
        self.crossatt = CrossAttention(dim_model, num_head, attn_pdrop, resid_pdrop)
        self.mlp_pro = nn.Sequential(nn.Linear(dim_model, block_exp * dim_model),
                                     nn.GELU(),
                                     nn.Linear(block_exp * dim_model, dim_model),
                                     nn.Dropout(resid_pdrop),
                                     )
        self.mlp_drg = nn.Sequential(nn.Linear(dim_model, block_exp * dim_model),
                                    nn.GELU(),
                                    nn.Linear(block_exp * dim_model, dim_model),
                                    nn.Dropout(resid_pdrop),
                                    )

        self.LN1 = nn.LayerNorm(dim_model)
        self.LN2 = nn.LayerNorm(dim_model)

        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        self.coefficient5 = LearnableCoefficient()
        self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()

    def forward(self, x_p, x_d, p_mask=None, d_mask=None):
        """
        Args:
            x (tuple): input features (x_vis, x_ir)
                x_vis: [batch, seq_len, dim_model] - visual features
                x_ir: [batch, seq_len, dim_model] - infrared features
            vis_mask: [batch, seq_len] - mask for visual features, True indicates padding
            ir_mask: [batch, seq_len] - mask for infrared features, True indicates padding
        Return:
            output (tuple): (out_vis, out_ir), each with shape [batch, seq_len, dim_model]
        """
        x_pro = x_p
        x_drg = x_d
        # assert x_pro.shape[0] == x_drg.shape[0]

        for _ in range(self.loops):
            pro_out, drg_out = self.crossatt(x_pro, x_drg, p_mask, d_mask)
            pro_att_out = self.coefficient1(x_pro) + self.coefficient2(pro_out)
            drg_att_out = self.coefficient3(x_drg) + self.coefficient4(drg_out)
            x_pro = self.coefficient5(pro_att_out) + self.coefficient6(self.mlp_pro(self.LN1(pro_att_out)))
            x_drg = self.coefficient7(drg_att_out) + self.coefficient8(self.mlp_drg(self.LN2(drg_att_out)))

        return x_pro, x_drg

class ICFE(nn.Module):
    def __init__(self, dim_model, num_head=4, block_exp=4, n_layer=1, loops_num=None, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super(ICFE, self).__init__()

        self.n_embd = dim_model

        self.apply(self._init_weights)

        self.crosstransformer = nn.ModuleList([
            CrossTransformerBlock(dim_model, num_head, attn_pdrop, resid_pdrop, loops_num, block_exp=block_exp) 
            for layer in range(n_layer)
        ])

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x_pro, x_drg, p_mask=None, d_mask=None):
        """
        Args:
            x_pro: [batch, seq_len, dim_model] - protein features
            x_drg: [batch, seq_len, dim_model] - drug features
            p_mask: [batch, seq_len] - mask for protein features, True indicates padding
            d_mask: [batch, seq_len] - mask for drug features, True indicates padding
        Return:
            out_pro: [batch, seq_len, dim_model] - fused protein features
            out_drg: [batch, seq_len, dim_model] - fused drug features
        """
        for layer in self.crosstransformer:
            x_pro, x_drg = layer(x_pro, x_drg, p_mask, d_mask)

        return x_pro, x_drg


class MHSA(nn.Module):
    def __init__(self, dim_model=128, n_layer=1, n_head=4, drop_rate=0.1):
        super(MHSA, self).__init__()
        self.n_layer = n_layer
        self.attn_layers = nn.ModuleList([
            CrossTransformerLayer(dim_model=dim_model, n_head=n_head, drop_rate=drop_rate)
            for _ in range(n_layer)
        ])

    def forward(self, x, x_mask):
        x_attn = x
        for attn_layer in self.attn_layers:
            x_attn = attn_layer(x_attn, x_attn, x_mask)
        
        return x_attn 

class MaskedAveragePooling(nn.Module):
    def __init__(self, keepdim=False):
        super(MaskedAveragePooling, self).__init__()
        self.keepdim = keepdim

    def forward(self, x, mask):
        mask_expanded = mask.unsqueeze(-1)
        x_masked = x.masked_fill(mask_expanded, 0)
        seq_lengths = mask.size(1) - mask.float().sum(dim=1, keepdim=True)
        seq_lengths = torch.clamp(seq_lengths, min=1)
        avg_pool = x_masked.sum(dim=1) / seq_lengths

        if self.keepdim:
            avg_pool = avg_pool.unsqueeze(1)

        return avg_pool


class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn_proj = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn_proj(x).squeeze(-1)        # [B, L]
        scores = scores.masked_fill(mask, -1e9)
        weights = torch.softmax(scores, dim=-1)        # [B, L]
        return (weights.unsqueeze(-1) * x).sum(dim=1)  # [B, dim]


class ConcatFusion(nn.Module):
    def __init__(self, dim: int = 128, use_attn_pool: bool = False):
        super().__init__()
        if use_attn_pool:
            self.drug_pool = AttentionPooling(dim)
            self.prot_pool = AttentionPooling(dim)
        else:
            self.drug_pool = MaskedAveragePooling()
            self.prot_pool = MaskedAveragePooling()

    def forward(self, v_d, v_p, v_d_mask, v_p_mask):
        d_pooled = self.drug_pool(v_d, v_d_mask)      # [B, dim]
        p_pooled = self.prot_pool(v_p, v_p_mask)      # [B, dim]
        return torch.cat([d_pooled, p_pooled], dim=-1) # [B, dim*2]


class DCPAttentionFusion(nn.Module):
    def __init__(self, dim: int = 128, n_head: int = 4):
        super().__init__()
        self.avgpool = MaskedAveragePooling()

        self.drug_query_proj = nn.Linear(dim, dim)
        self.protein_cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_head, batch_first=True
        )
        self.prot_ctx_norm = nn.LayerNorm(dim)

        self.prot_query_proj = nn.Linear(dim, dim)
        self.drug_cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_head, batch_first=True
        )
        self.drug_ctx_norm = nn.LayerNorm(dim)

    def forward(self, v_d, v_p, v_d_mask, v_p_mask):
        """
        Args:
            v_d:      [B, L_d, dim]
            v_p:      [B, L_p, dim]
            v_d_mask: [B, L_d] — True = padding
            v_p_mask: [B, L_p] — True = padding
        Returns:
            [B, dim*2]
        """
        d_global = self.avgpool(v_d, v_d_mask)                      # [B, dim]
        p_global = self.avgpool(v_p, v_p_mask)                      # [B, dim]

        # Drug-conditioned protein context: what protein residues matter for this substrate?
        q_d = self.drug_query_proj(d_global).unsqueeze(1)           # [B, 1, dim]
        p_ctx, _ = self.protein_cross_attn(
            q_d, v_p, v_p, key_padding_mask=v_p_mask
        )                                                            # [B, 1, dim]
        p_ctx = self.prot_ctx_norm(p_ctx.squeeze(1))                # [B, dim]

        # Protein-conditioned drug context: which drug atoms interact with the enzyme?
        q_p = self.prot_query_proj(p_global).unsqueeze(1)           # [B, 1, dim]
        d_ctx, _ = self.drug_cross_attn(
            q_p, v_d, v_d, key_padding_mask=v_d_mask
        )                                                            # [B, 1, dim]
        d_ctx = self.drug_ctx_norm(d_ctx.squeeze(1))                # [B, dim]

        return torch.cat([p_ctx, d_ctx], dim=-1)                    # [B, dim*2]




    