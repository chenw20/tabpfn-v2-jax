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


def export_for_jax(dataset_name='breast_cancer'):
    """Export all components needed for JAX inference."""
    
    if dataset_name == 'breast_cancer':
        print("Dataset: Breast Cancer (Classification)")
        X, y = load_breast_cancer(return_X_y=True)
        is_regression = False
        out_path = 'tabpfn_jax/jax_inference_bundle.pkl'
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
    elif dataset_name == 'boston':
        print("Dataset: Boston Housing (Regression)")
        # Boston housing is deprecated in sklearn 1.2+, use fetch_openml or alternative
        # Using a simple regression dataset generator or fetch_california_housing if boston fails
        try:
            from sklearn.datasets import load_boston
            X, y = load_boston(return_X_y=True)
        except ImportError:
            print("load_boston failed, trying fetch_california_housing...")
            from sklearn.datasets import fetch_california_housing
            X, y = fetch_california_housing(return_X_y=True)
            # Subsample for speed/memory if california
            X = X[:1000]
            y = y[:1000]
            
        is_regression = True
        out_path = 'tabpfn_jax/jax_inference_bundle_reg.pkl'
        from tabpfn import TabPFNRegressor
        clf = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    
    print(f"Setting up PyTorch TabPFN V2...")
    clf.device = 'cpu'
    
    # Fit
    clf.fit(X_train, y_train)
    
    # Move to CPU float
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
        'is_regression': is_regression,
    }
    print(f"Config: {export_data['config']}")
    
    # 2. CPU preprocessor (fitted sklearn pipeline)
    export_data['cpu_preprocessor'] = em.cpu_preprocessor
    export_data['gpu_preprocessor_std'] = em.config.outlier_removal_std
    
    # 3. Ensemble config info
    export_data['ensemble_config'] = {
        'feature_shift_count': em.config.feature_shift_count,
        'feature_shift_decoder': em.config.feature_shift_decoder,
        'class_permutation': getattr(em.config, 'class_permutation', None),
        'add_fingerprint_feature': em.config.add_fingerprint_feature,
    }
    
    # 4. X encoder weights (LinearInputEncoderStep)
    x_enc_layer = pt_model.encoder[-1].layer
    x_enc_w = x_enc_layer.weight.detach().cpu().numpy()  # [192, 4]
    export_data['x_encoder_weight'] = x_enc_w.T  # [4, 192] for JAX Dense
    
    if x_enc_layer.bias is not None:
        export_data['x_encoder_bias'] = x_enc_layer.bias.detach().cpu().numpy()
        print(f"Exported x_encoder_bias: {export_data['x_encoder_bias'].shape}")
    else:
        export_data['x_encoder_bias'] = None
        print("x_encoder has no bias")
    
    # 5. Y encoder weights
    # For regression, y encoder might be different or same structure
    y_enc_layer = pt_model.y_encoder[-1].layer
    y_enc_w = y_enc_layer.weight.detach().cpu().numpy()
    y_enc_b = y_enc_layer.bias.detach().cpu().numpy()
    export_data['y_encoder_weight'] = y_enc_w.T
    export_data['y_encoder_bias'] = y_enc_b
    print(f"y_encoder weight shape: {y_enc_w.shape}")
    
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
    
    if is_regression:
        export_data['n_classes'] = 0 # Not used for regression
        export_data['softmax_temperature'] = 1.0 # Not used
        export_data['classes'] = []
        
        # Regression specific metadata
        if hasattr(clf, 'y_train_mean_'):
            export_data['y_train_mean'] = clf.y_train_mean_
            export_data['y_train_std'] = clf.y_train_std_
            print(f"Exported y_train_mean: {clf.y_train_mean_}, std: {clf.y_train_std_}")
        
        if hasattr(clf, 'znorm_space_bardist_'):
             export_data['znorm_borders'] = clf.znorm_space_bardist_.borders.cpu().numpy()
             print(f"Exported znorm_borders: {export_data['znorm_borders'].shape}")
             
    else:
        export_data['n_classes'] = clf.n_classes_
        export_data['softmax_temperature'] = clf.softmax_temperature_
        export_data['classes'] = clf.classes_
    
    # 9. Reference Output and Data for Verification
    # Run PyTorch inference on X_test to get ground truth decoder logits for validation
    print("\nRunning PyTorch inference for verification reference...")
    pt_model.eval()
    
    # Use the em to transform X_test
    X_test_pp = em.transform_X_test(X_test)
    X_train_pp = em.X_train
    y_train_pp = em.y_train
    
    X_full = torch.cat([
        torch.as_tensor(X_train_pp, dtype=torch.float32),
        torch.as_tensor(X_test_pp, dtype=torch.float32)
    ], dim=0).unsqueeze(1)
    
    y_full = torch.as_tensor(y_train_pp, dtype=torch.float32)
    
    # Run model
    from tabpfn.inference import _maybe_run_gpu_preprocessing
    from tabpfn.preprocessing.datamodel import FeatureModality
    
    with torch.no_grad():
        X_full_gpu = _maybe_run_gpu_preprocessing(
            X_full, em.gpu_preprocessor, num_train_rows=len(X_train_pp)
        )
        cat_inds = [em.feature_schema.indices_for(FeatureModality.CATEGORICAL)]
        
        pt_output = pt_model(
            X_full_gpu, y_full,
            only_return_standard_out=True,
            categorical_inds=cat_inds
        )
    
    export_data['verification_data'] = {
        'X_test': X_test,
        'pytorch_output': pt_output.cpu().numpy()  # [seq, 1, 10]
    }
    
    # Export RemoveEmptyFeatures mask from MODEL encoder
    if hasattr(pt_model, 'encoder'):
        encoder = pt_model.encoder
        # Step 0 is typically RemoveEmptyFeaturesEncoderStep
        if len(encoder) > 0:
            remove_empty_step = encoder[0]
            print(f"DEBUG: Encoder Step 0 type: {type(remove_empty_step)}")
            if hasattr(remove_empty_step, 'column_selection_mask') and remove_empty_step.column_selection_mask is not None:
                 mask = remove_empty_step.column_selection_mask
                 export_data['feature_selection_mask'] = mask.cpu().numpy()
                 print(f"DEBUG: Exported feature_selection_mask shape: {export_data['feature_selection_mask'].shape}")
            else:
                 export_data['feature_selection_mask'] = None
                 print("DEBUG: No feature_selection_mask found in Step 0")
    
    # Check X_train shape
    print(f"DEBUG: export_data['X_train_preprocessed'] shape: {export_data['X_train_preprocessed'].shape}")
    
    # Check what CPU preprocessor produces
    X_sample = X_train[:5]
    X_sample_pp = em.cpu_preprocessor.transform(X_sample)
    if hasattr(X_sample_pp, 'X'):
        X_sample_pp = X_sample_pp.X
    print(f"DEBUG: cpu_preprocessor output shape for 5 samples: {X_sample_pp.shape}")
    
    # Save
    with open(out_path, 'wb') as f:
        pickle.dump(export_data, f)
    
    print(f"\nSaved complete inference bundle to {out_path}")
    print(f"Bundle contains: {list(export_data.keys())}")
    
    return export_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='breast_cancer', choices=['breast_cancer', 'boston'], help='Dataset to export')
    args = parser.parse_args()
    
    export_for_jax(dataset_name=args.dataset)
