"""
TODO: transformer train
Author: Redal
Date: 2025/03/21
Homepage: https://github.com/Redal/Multimodal-Research.git
"""
import argparse
import torch
from torch import nn
from models.model.transformer import Transformer


def config():
    parser = argparse.ArgumentParser(description='Transformer Inital Parameters',
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src_pad_idx', type=int, default=0, help='source padding index')
    parser.add_argument('--trg_pad_idx', type=int, default=0, help='target padding index')
    parser.add_argument('--trg_sos_idx', type=int, default=0, help='target start of sentence index')
    parser.add_argument('--enc_voc_size', type=int, default=1000, help='encoder vocabulary size')
    parser.add_argument('--dec_voc_size', type=int, default=1000, help='decoder vocabulary size')
    parser.add_argument('--d_model', type=int, default=512, help='model dimension')
    parser.add_argument('--n_heads', type=int, default=2, help='number of heads')
    parser.add_argument('--max_len', type=int, default=100, help='maximum length')
    parser.add_argument('--ffn_hidden', type=int, default=2048, help='feed forward network hidden dimension')
    parser.add_argument('--n_layers', type=int, default=6, help='number of layers')
    parser.add_argument('--drop_prob', type=float, default=0.1, help='dropout probability')
    parser.add_argument('--device', type=str, default=torch.device('cuda:0'), help='device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = config()
    transformer = Transformer(args.src_pad_idx, args.trg_pad_idx,
                            args.trg_sos_idx, args.enc_voc_size, 
                            args.dec_voc_size, args.d_model, args.n_heads,
                            args.max_len, args.ffn_hidden, args.n_layers,
                            args.drop_prob, args.device).to(args.device)
    print(transformer)