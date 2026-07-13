import torch.nn as nn
import torch.nn.functional as F
import torch
from module.Encoder import *
from module.CN import *

class CatESO(nn.Module):
    def __init__(self, **config):
        super(CatESO, self).__init__()
        """
        drug: features related to substrate;
        protein: features related to enzyme;     
        """
        
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        drug_padding = config["DRUG"]["PADDING"]

        activation_str = config["DRUG"].get("ACTIVATION", "relu")
        activation_map = {
            "relu": F.relu,
            "leaky_relu": F.leaky_relu,
            "elu": F.elu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            None: F.relu  
        }
        drug_activation = activation_map.get(activation_str, F.relu)

        protein_in_dim = config["PROTEIN"]["IN_DIM"]
        protein_hidden_dim = config["PROTEIN"]["HIDDEN_DIM"]
        protein_target_dim = config["PROTEIN"]["TARGET_DIM"]
        
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        out_binary = config["DECODER"]["BINARY"]
        
        self.icfe_dim = config["ICFE"]["DIM"]
        self.icfe_num_head = config["ICFE"]["NUM_HEAD"]
        self.icfe_block_exp = config["ICFE"]["BLOCK_EXP"]
        self.icfe_n_layer = config["ICFE"]["N_LAYER"]
        self.icfe_embd_pdrop = config["ICFE"]["EMBD_PDROP"]
        self.icfe_attn_pdrop = config["ICFE"]["ATTN_PDROP"]
        self.icfe_resid_pdrop = config["ICFE"]["RESID_PDROP"]
        self.icfe_loops_num = config["ICFE"]["LOOPS_NUM"]
        self.icfe_flag = config["ICFE"]["IS_ICFE"]
        
        self.mhsa_dim = config["MHSA"]["DIM"]
        self.mhsa_n_layer = config["MHSA"]["N_LAYER"]
        self.mhsa_n_head = config["MHSA"]["N_HEAD"]
        self.mhsa_drop_rate = config["MHSA"]["DROP_RATE"]
        self.mhsa_flag = config["MHSA"]["IS_MHSA"]
        
        self.mcdc_kernel_list = config["MCDC"]["KERNEL_LIST"]
        self.mcdc_K = config["MCDC"]["K"]
        self.mcdc_flag = config["MCDC"]["IS_MCDC"]

        self.fusion_is_dcpa   = config["FUSION"]["IS_DCPA"]
        self.fusion_is_concat = config["FUSION"]["IS_CONCAT"]
        self.fusion_attn_pool = config["FUSION"]["USE_ATTN_POOL"]
        self.fusion_n_head    = config["FUSION"]["N_HEAD"]
        
        self.drug_extractor = Encoder_drug(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats,
                                           activation=drug_activation)  
        self.protein_extractor = Encoder_protein(protein_in_dim, protein_hidden_dim, protein_target_dim)

        if self.mhsa_flag:
            self.mhsa = MHSA(dim_model=self.mhsa_dim, n_layer=self.mhsa_n_layer, 
                          n_head=self.mhsa_n_head, drop_rate=self.mhsa_drop_rate)
        else:
            self.mhsa = None
            
        if self.icfe_flag:
            self.icfe = ICFE(dim_model=self.icfe_dim, num_head=self.icfe_num_head, block_exp=self.icfe_block_exp,
                                n_layer=self.icfe_n_layer, loops_num=self.icfe_loops_num, embd_pdrop=self.icfe_embd_pdrop, attn_pdrop=self.icfe_attn_pdrop,
                                resid_pdrop=self.icfe_resid_pdrop)
        else:
            self.icfe = None

        if self.mcdc_flag:
            self.mcdc = MCDC(channels=self.icfe_dim, kernel_list=self.mcdc_kernel_list, K=self.mcdc_K, use_bn=True, activation='relu')
        else:
            self.mcdc = None

        if self.fusion_is_dcpa:
            self.fusion = DCPAttentionFusion(dim=self.icfe_dim, n_head=self.fusion_n_head)
        elif self.fusion_is_concat:
            self.fusion = ConcatFusion(dim=self.icfe_dim, use_attn_pool=self.fusion_attn_pool)
        else:
            self.fusion = SimpleFusion()

        decoder_deep = config["DECODER"]["DEEP"]
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary, deep=decoder_deep)

    def forward(self, v_d, v_p, v_d_mask, v_p_mask):
        v_d = self.drug_extractor(v_d)
        v_p = self.protein_extractor(v_p)

        if self.mhsa is not None:
            v_d = self.mhsa(v_d, v_d_mask)
            
        if self.mcdc is not None:
            v_p = self.mcdc(v_p, v_d, v_p_mask, v_d_mask)
        
        if self.icfe is not None:
            v_p, v_d = self.icfe(v_p, v_d, v_p_mask, v_d_mask)

        f = self.fusion(v_d, v_p, v_d_mask, v_p_mask)
        
        score = self.mlp_classifier(f)

        return v_d, v_p, f, score

class MLPDecoder(nn.Module):
    """
    MLP regression head.

    deep=False (default): original 2-layer design, BN + Tanh.
    deep=True:            3-layer design, LayerNorm + GELU.
                          LayerNorm is more stable than BN for regression
                          and does not depend on batch statistics.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1, deep=False):
        super(MLPDecoder, self).__init__()
        if deep:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Linear(out_dim, binary),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.Tanh(),
                nn.Linear(out_dim, binary),
            )

    def forward(self, x):
        return self.net(x)

class SimpleFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = MaskedAveragePooling()

    def forward(self, v_d, v_p, v_d_mask, v_p_mask):
        tensor1_pooled = self.avgpool(v_d, v_d_mask)
        tensor2_pooled = self.avgpool(v_p, v_p_mask)

        concatenated = tensor1_pooled + tensor2_pooled

        return concatenated