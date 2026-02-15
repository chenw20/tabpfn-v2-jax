"""
Save all components needed for standalone JAX TabPFN inference:
- Fitted CPU preprocessor (sklearn-based)
- Model encoder weights (x_encoder, y_encoder linear projections)
- Transformer + decoder weights (already saved)
- Feature positional embedding

Run this script ONCE to export everything, then use jax_tabpfn_predict.py
for pure JAX inference without needing PyTorch at all.
"""
import sys
import os
import torch
import numpy as np
import pickle

sys.path.insert(0, os.path.abspath("TabPFN/src"))

from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def export_for_jax(mode='classifier'):
    """Export all components needed for JAX inference."""
    
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    
    print(f"Setting up PyTorch TabPFN V2 {mode}...")
    clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
    clf.device = 'cpu'
    clf.fit(X_train, y_train)
    for i, m in enumerate(clf.models_):
        clf.models_[i] = m.cpu().float()
    
    pt_model = clf.models_[0]
    config = clf.configs_[0]
    em = clf.executor_.ensemble_members[0]
    
    export_data = {}
    
    # 1. Config info
    export_data['config'] = {
        'emsize': config.emsize,
        'nhead': config.nhead,
        'nlayers': config.nlayers,
        'nhid_factor': config.nhid_factor,
        'features_per_group': config.features_per_group,
        'n_out': pt_model.n_out,
    }
    print(f"Config: {export_data['config']}")
    
    # 2. CPU preprocessor (fitted sklearn pipeline)
    export_data['cpu_preprocessor'] = em.cpu_preprocessor
    export_data['gpu_preprocessor_std'] = em.config.outlier_removal_std
    
    # 3. Ensemble config info
    export_data['ensemble_config'] = {
        'feature_shift_count': em.config.feature_shift_count,
        'feature_shift_decoder': em.config.feature_shift_decoder,
        'class_permutation': em.config.class_permutation,
        'add_fingerprint_feature': em.config.add_fingerprint_feature,
    }
    
    # 4. X encoder weights (LinearInputEncoderStep)
    x_enc_w = pt_model.encoder[-1].layer.weight.detach().cpu().numpy()  # [192, 4]
    export_data['x_encoder_weight'] = x_enc_w.T  # [4, 192] for JAX Dense
    # x encoder has no bias
    
    # 5. Y encoder weights
    y_enc_w = pt_model.y_encoder[-1].layer.weight.detach().cpu().numpy()  # [192, 2]
    y_enc_b = pt_model.y_encoder[-1].layer.bias.detach().cpu().numpy()  # [192]
    export_data['y_encoder_weight'] = y_enc_w.T  # [2, 192]
    export_data['y_encoder_bias'] = y_enc_b
    
    # 6. Feature positional embedding
    from pathlib import Path
    col_emb_path = Path("TabPFN/src/tabpfn/architectures/base/tabpfn_col_embedding.pt")
    col_embedding = torch.load(col_emb_path, map_location='cpu').numpy()
    export_data['col_embedding'] = col_embedding
    
    feat_pos_w = pt_model.feature_positional_embedding_embeddings.weight.detach().cpu().numpy()
    feat_pos_b = pt_model.feature_positional_embedding_embeddings.bias.detach().cpu().numpy()
    export_data['feat_pos_emb_weight'] = feat_pos_w.T  # [48, 192]
    export_data['feat_pos_emb_bias'] = feat_pos_b
    
    # 7. Transformer + decoder weights (reuse existing conversion)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from convert_weights import convert_state_dict
    
    state_dict = pt_model.state_dict()
    transformer_params = convert_state_dict(state_dict, n_layers=config.nlayers)
    export_data['transformer_params'] = transformer_params
    
    # 8. Training data (needed at inference time — TabPFN is in-context learning)
    export_data['X_train_preprocessed'] = em.X_train
    export_data['y_train'] = em.y_train
    export_data['n_classes'] = clf.n_classes_
    
    # 9. Softmax temperature
    export_data['softmax_temperature'] = clf.softmax_temperature_
    
    # 10. Label encoder for mapping back to original labels
    export_data['classes'] = clf.classes_
    
    # Save
    out_path = 'tabpfn_jax/jax_inference_bundle.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(export_data, f)
    
    print(f"\nSaved complete inference bundle to {out_path}")
    print(f"Bundle contains: {list(export_data.keys())}")
    
    # Verify x encoder
    print(f"\nx_encoder weight shape: {x_enc_w.shape}")
    print(f"y_encoder weight shape: {y_enc_w.shape}")
    print(f"X_train_preprocessed shape: {em.X_train.shape}")
    print(f"y_train shape: {em.y_train.shape}")
    
    return export_data


if __name__ == "__main__":
    export_for_jax()
