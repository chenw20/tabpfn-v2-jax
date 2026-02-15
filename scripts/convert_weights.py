
import sys
import os
import torch
import jax.numpy as jnp
import numpy as np
import pickle
from pathlib import Path
import argparse

# Add the TabPFN repo to path
sys.path.insert(0, os.path.abspath("TabPFN/src"))
from tabpfn.constants import ModelVersion


def load_v2_model(mode='classifier'):
    """Load TabPFN V2 model (classifier or regressor)."""
    print(f"Loading TabPFN V2 {mode} model...")
    if mode == 'classifier':
        from tabpfn.classifier import TabPFNClassifier
        est = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
    else:
        from tabpfn.regressor import TabPFNRegressor
        est = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
    est._initialize_model_variables()
    return est.models_[0], est.configs_[0]


def convert_state_dict(state_dict, n_layers=12):
    params = {}
    
    # Feature positional embedding
    w = state_dict['feature_positional_embedding_embeddings.weight'].cpu().numpy()
    b = state_dict['feature_positional_embedding_embeddings.bias'].cpu().numpy()
    params['feature_positional_embedding_embeddings'] = {'kernel': w.T, 'bias': b}
    
    # Decoder
    w = state_dict['decoder_dict.standard.0.weight'].cpu().numpy()
    b = state_dict['decoder_dict.standard.0.bias'].cpu().numpy()
    params['decoder_linear1'] = {'kernel': w.T, 'bias': b}
    
    w = state_dict['decoder_dict.standard.2.weight'].cpu().numpy()
    b = state_dict['decoder_dict.standard.2.bias'].cpu().numpy()
    params['decoder_linear2'] = {'kernel': w.T, 'bias': b}
    
    # Transformer layers
    for i in range(n_layers):
        layer_params = {}
        prefix = f'transformer_encoder.layers.{i}.'
        
        # self_attn_between_features
        w_qkv = state_dict[prefix + 'self_attn_between_features._w_qkv'].cpu().numpy()
        w_qkv = np.transpose(w_qkv, (3, 0, 1, 2)).reshape(w_qkv.shape[3], -1)
        
        w_out = state_dict[prefix + 'self_attn_between_features._w_out'].cpu().numpy()
        w_out = w_out.reshape(-1, w_out.shape[-1])
        
        layer_params['self_attn_between_features'] = {
            'w_qkv_kernel': w_qkv, 
            'w_out': {'kernel': w_out}
        }
        
        # self_attn_between_items
        w_qkv = state_dict[prefix + 'self_attn_between_items._w_qkv'].cpu().numpy()
        w_qkv = np.transpose(w_qkv, (3, 0, 1, 2)).reshape(w_qkv.shape[3], -1)
        
        w_out = state_dict[prefix + 'self_attn_between_items._w_out'].cpu().numpy()
        w_out = w_out.reshape(-1, w_out.shape[-1])
        
        layer_params['self_attn_between_items'] = {
            'w_qkv_kernel': w_qkv, 
            'w_out': {'kernel': w_out}
        }
        
        # MLP
        w = state_dict[prefix + 'mlp.linear1.weight'].cpu().numpy()
        layer_params['mlp'] = {'linear1': {'kernel': w.T}}
        w = state_dict[prefix + 'mlp.linear2.weight'].cpu().numpy()
        layer_params['mlp']['linear2'] = {'kernel': w.T}
        
        params[f'layers_{i}'] = layer_params
        
    return params


def load_col_embedding():
    path = Path("TabPFN/src/tabpfn/architectures/base/tabpfn_col_embedding.pt")
    if not path.exists():
        path = Path("tabpfn_concise_v2/src/tabpfn/architectures/base/tabpfn_col_embedding.pt")
    return torch.load(path, map_location='cpu').numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['classifier', 'regressor'], default='classifier')
    args = parser.parse_args()
    
    model, config = load_v2_model(args.mode)
    state_dict = model.state_dict()
    
    print(f"n_out: {model.n_out}")
    print(f"nlayers: {config.nlayers}")
    
    jax_params = convert_state_dict(state_dict, n_layers=config.nlayers)
    col_embedding = load_col_embedding()
    
    output = {
        'params': jax_params, 
        'col_embedding': col_embedding,
        'n_out': model.n_out,
        'mode': args.mode,
    }
    
    out_file = f'tabpfn_jax/jax_params_{args.mode}.pkl'
    with open(out_file, 'wb') as f:
        pickle.dump(output, f)
        
    print(f"Saved {out_file}")
    
    # Also save as default if classifier
    if args.mode == 'classifier':
        with open('tabpfn_jax/jax_params.pkl', 'wb') as f:
            pickle.dump(output, f)
        print("Also saved as jax_params.pkl")
