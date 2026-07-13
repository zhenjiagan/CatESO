# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import argparse

from torch import nn
import torch.nn.functional as F
import esm
import os
from tqdm import tqdm

# Location of the ESM-2 3B weight (embedding.py uses it to extract protein residue
# representations). Priority: --esm_ckpt CLI arg > the default path below.
# The default is derived from the ESM_CKPT_DIR env var (shared with main_optimize.py
# and the sbatch scripts); weights live under $ESM_CKPT_DIR/hub/checkpoints/.
# See the "Weights" section of the README for how to download them.
DEFAULT_ESM_CKPT = os.path.join(
    os.environ.get("ESM_CKPT_DIR", "/scratch/zg2470/code/esm/ckpt"),
    "hub", "checkpoints", "esm2_t36_3B_UR50D.pt",
)

class ESM_model(nn.Module):
    def __init__(self, ckpt_path=DEFAULT_ESM_CKPT):
        super(ESM_model, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"ESM-2 weight not found: {ckpt_path}\n"
                "Please download esm2_t36_3B_UR50D.pt first (see the 'Weights' section "
                "of the README), or specify its location via --esm_ckpt / ESM_CKPT_DIR."
            )
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(ckpt_path)
        self.model = self.model.to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()
    def forward(self, data):
        tmp = []
        for seq in data:
            tmp.append(('',seq))
        data = tmp
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens[:, 0:1024]
        
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[36], return_contacts=False)
        token_representations = results["representations"][36]
        return token_representations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Embedding extraction")
    parser.add_argument('--feat_dir', required=True, help="path to embeddings", type=str)
    parser.add_argument('--data_dir', required=True, help="path to data", type=str)
    parser.add_argument('--esm_ckpt', default=DEFAULT_ESM_CKPT, type=str,
                        help="path to esm2_t36_3B_UR50D.pt "
                             "(default: $ESM_CKPT_DIR/hub/checkpoints/esm2_t36_3B_UR50D.pt)")
    args = parser.parse_args()

    dataset_list = ['CatPred']
    for dataset in dataset_list:
        print(f'Embedding extraction for: {dataset}')
        path = f'{args.data_dir}/{dataset}/'
    
        train_df = pd.read_csv(os.path.join(path, 'train.csv'))
        test_df = pd.read_csv(os.path.join(path, 'test.csv'))
        val_df = pd.read_csv(os.path.join(path, 'val.csv'))
    
        df = pd.concat([train_df, test_df, val_df], ignore_index=True)
    
        protein_set = df['Protein'].drop_duplicates().reset_index(drop=True)
        protein_index = {protein: index for index, protein in enumerate(protein_set)}
    
        df['Protein_index'] = df['Protein'].map(protein_index)
    
        model = ESM_model(args.esm_ckpt)
    
        dataset_dir = f'{args.feat_dir}/{dataset}'
        feat_dir = os.path.join(dataset_dir, 'esm')

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        if not os.path.exists(feat_dir):
            os.makedirs(feat_dir)
    
        for seq, index in tqdm(protein_index.items()):
    
            feat = model([seq])
            feat = feat.squeeze(dim=0)
            feat = feat[1:len(seq)+1, :]
            feat = feat.cpu().detach()
            torch.save(feat, os.path.join(feat_dir, f'{index}.pt'))
    
        train_df['Protein_index'] = train_df['Protein'].map(protein_index)
        val_df['Protein_index'] = val_df['Protein'].map(protein_index)
        test_df['Protein_index'] = test_df['Protein'].map(protein_index)
    
        train_df['Protein_Path'] = train_df['Protein_index'].apply(lambda x: f'{dataset_dir}/esm/{x}.pt')
        val_df['Protein_Path'] = val_df['Protein_index'].apply(lambda x: f'{dataset_dir}/esm/{x}.pt')
        test_df['Protein_Path'] = test_df['Protein_index'].apply(lambda x: f'{dataset_dir}/esm/{x}.pt')
    
        train_df.to_csv(os.path.join(dataset_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(dataset_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(dataset_dir, 'test.csv'), index=False)



