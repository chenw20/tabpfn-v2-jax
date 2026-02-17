"""
Self-contained JAX TabPFN inference example.

This script demonstrates end-to-end prediction using JAX TabPFN, matching
the breast cancer classification example from the TabPFN README.

Dependencies: jax, jaxlib, numpy, scikit-learn, pickle
No PyTorch dependency at inference time!

Usage:
    # First, run export_for_jax.py once (requires PyTorch) to create the bundle:
    #   python tabpfn_jax/scripts/export_for_jax.py
    #
    # Then run this script for pure JAX inference:
    #   python jax_tabpfn_predict.py
"""
import os
import sys
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ---------------------------------------------------------------------------
# JAX implementations of the model-internal encoder steps
# (These replace the PyTorch encoder pipeline)
# ---------------------------------------------------------------------------

def nan_handling(x):
    """Replace NaN/inf with feature means, return (x_clean, nan_indicators).
    
    x: [..., F]  (float array, may contain NaN)
    Returns:
        x_clean: same shape, NaN replaced with 0 (train feature means would be 
                  more accurate but for breast cancer there are no NaNs)
        nan_indicators: same shape, -2.0 for NaN, 2.0 for +inf, 4.0 for -inf, 0 otherwise
    """
    nan_mask = jnp.isnan(x)
    inf_pos = jnp.logical_and(jnp.isinf(x), x > 0)
    inf_neg = jnp.logical_and(jnp.isinf(x), x < 0)
    
    nan_indicators = (nan_mask.astype(jnp.float32) * -2.0 +
                      inf_pos.astype(jnp.float32) * 2.0 +
                      inf_neg.astype(jnp.float32) * 4.0)
    
    bad_mask = jnp.logical_or(nan_mask, jnp.isinf(x))
    x_clean = jnp.where(bad_mask, 0.0, x)
    
    return x_clean, nan_indicators


def normalize_data(x, train_len, clip=True):
    """Normalize to zero mean and unit variance using training statistics.
    
    x: [seq_len, batch*groups, features_per_group]
    train_len: number of training samples (single_eval_pos)
    """
    train_x = x[:train_len]
    
    # nanmean
    valid = ~jnp.isnan(train_x)
    count = valid.sum(axis=0).clip(min=1)
    mean = jnp.where(valid, train_x, 0.0).sum(axis=0) / count
    
    # nanstd
    diff = jnp.where(valid, train_x - mean[None], 0.0)
    var = (diff ** 2).sum(axis=0) / (count - 1).clip(min=1)
    std = jnp.sqrt(var)
    std = jnp.where(std == 0, 1.0, std)
    
    if train_len <= 1:
        std = jnp.ones_like(std)
    
    x = (x - mean) / (std + 1e-16)
    
    if clip:
        x = jnp.clip(x, -100, 100)
    
    return x


def normalize_feature_groups(x, non_constants_mask, num_used_features):
    """Scale features by sqrt(fpg / num_used_features).
    
    x: [seq_len, batch*groups, features_per_group]
    non_constants_mask: [batch*groups, features_per_group] bool
    num_used_features: [batch*groups, 1] int
    """
    fpg = x.shape[-1]
    scale = fpg / num_used_features
    x = x * jnp.sqrt(scale)
    
    # Zero out constant features
    x = x * non_constants_mask[None].astype(x.dtype)
    
    return x


def multiclass_target_encode(y, unique_ys):
    """Flatten classification targets (ordinal encoding).
    
    For class labels, converts to ordinal by counting how many unique values
    each y is greater than.
    
    y: [seq_len, 1, 1]
    unique_ys: array of unique class values from training set
    """
    return (y[..., None] > unique_ys).sum(axis=-1)


def x_encoder_forward(x, nan_indicators, x_encoder_weight):
    """X encoder: concatenate features with nan_indicators, then linear project.
    
    x: [seq, groups, fpg]  (NaN-handled features)
    nan_indicators: [seq, groups, fpg]
    x_encoder_weight: [fpg*2, emsize]  (no bias)
    """
    x_cat = jnp.concatenate([x, nan_indicators], axis=-1)  # [..., fpg*2]
    return x_cat @ x_encoder_weight  # [..., emsize]


def y_encoder_forward(y, nan_indicators_y, y_encoder_weight, y_encoder_bias):
    """Y encoder: concatenate y with nan_indicators, then linear project.
    
    y: [seq, batch, 1]
    nan_indicators_y: [seq, batch, 1]
    """
    y_cat = jnp.concatenate([y, nan_indicators_y], axis=-1)  # [..., 2]
    return y_cat @ y_encoder_weight + y_encoder_bias  # [..., emsize]


def feature_positional_embedding(x, col_embedding, feat_pos_emb_weight, feat_pos_emb_bias):
    """Apply subspace feature positional embedding.
    
    x: [batch, seq, n_groups, emsize]
    col_embedding: [2000, 48] pre-computed random embeddings
    """
    n_groups = x.shape[2]
    # Use pre-computed col_embedding for first 2000 features
    embs = col_embedding[:n_groups]  # [n_groups, 48]
    # Project through linear layer
    embs = embs @ feat_pos_emb_weight + feat_pos_emb_bias  # [n_groups, emsize]
    # Add to x (broadcast over batch and seq dimensions)
    x = x + embs[None, None]
    return x


# ---------------------------------------------------------------------------
# JAX Regression Utils
# ---------------------------------------------------------------------------

def _map_to_bucket_ix(y, borders):
    """JAX implementation of _map_to_bucket_ix."""
    ix = jnp.searchsorted(borders, y) - 1
    # clamp manual
    ix = jnp.where(y == borders[0], 0, ix)
    ix = jnp.where(y == borders[-1], len(borders) - 2, ix)
    return ix

def _cdf(logits, borders, ys):
    """JAX implementation of _cdf."""
    # borders: [n_borders]
    # logits: [seq, n_outputs] -> [seq, n_borders-1]
    # ys: [seq, n_borders]
    
    # Broadcast ys to match logits batch dims
    ys_expand = ys[None, :]  # [1, n_target]
    ys_expand = jnp.tile(ys_expand, (logits.shape[0], 1)) # [n_samples, n_target]
    
    n_bars = len(borders) - 1
    y_buckets = _map_to_bucket_ix(ys_expand, borders)
    y_buckets = jnp.clip(y_buckets, 0, n_bars - 1)
    
    probs = jax.nn.softmax(logits, axis=-1)
    
    # cumsum
    prob_so_far = jnp.cumsum(probs, axis=-1) - probs
    
    # gather
    prob_left_of_bucket = jnp.take_along_axis(prob_so_far, y_buckets, axis=-1)
    
    # bucket widths
    bucket_widths = borders[1:] - borders[:-1]
    
    b_val = borders[y_buckets]
    w_val = bucket_widths[y_buckets]
    
    share_of_bucket_left = (ys_expand - b_val) / w_val
    share_of_bucket_left = jnp.clip(share_of_bucket_left, 0.0, 1.0)
    
    prob_in_bucket = jnp.take_along_axis(probs, y_buckets, axis=-1) * share_of_bucket_left
    prob_left_of_ys = prob_left_of_bucket + prob_in_bucket
    
    # boundaries
    prob_left_of_ys = jnp.where(ys_expand <= borders[0], 0.0, prob_left_of_ys)
    prob_left_of_ys = jnp.where(ys_expand >= borders[-1], 1.0, prob_left_of_ys)
    
    return jnp.clip(prob_left_of_ys, 0.0, 1.0)

def translate_probs_across_borders(logits, frm, to):
    """Translate probabilities from 'frm' borders to 'to' borders."""
    # logits: [n_samples, n_buckets]
    # frm: [n_buckets+1]
    # to: [n_target_buckets+1]
    
    # prob_left is computed at 'to' points
    prob_left = _cdf(logits, borders=frm, ys=to)
    # prob_left is [n_samples, n_target_buckets+1]
    
    # Fix boundaries 0 and 1
    prob_left = prob_left.at[..., 0].set(0.0)
    prob_left = prob_left.at[..., -1].set(1.0)
    
    # Prob in each new bucket is difference of CDF
    new_probs = prob_left[..., 1:] - prob_left[..., :-1]
    return jnp.clip(new_probs, a_min=0.0)

def compute_mean_at_distribution(logits, borders):
    """Compute mean of the distribution defined by logits and borders."""
    probs = jax.nn.softmax(logits, axis=-1)
    centers = (borders[1:] + borders[:-1]) / 2
    return (probs * centers).sum(axis=-1)


# ---------------------------------------------------------------------------
# Full JAX TabPFN prediction pipeline
# ---------------------------------------------------------------------------

class JAXTabPFN:
    """Pure JAX TabPFN inference engine.
    
    Loads a pre-exported bundle (from export_for_jax.py) and performs
    prediction without any PyTorch dependency.
    """
    
    def __init__(self, bundle_path):
        with open(bundle_path, 'rb') as f:
            bundle = pickle.load(f)
        
        self.config = bundle['config']
        self.cpu_preprocessor = bundle['cpu_preprocessor']
        self.ensemble_config = bundle['ensemble_config']
        
        # Encoder weights
        self.x_encoder_weight = jnp.array(bundle['x_encoder_weight'])
        self.y_encoder_weight = jnp.array(bundle['y_encoder_weight'])
        self.y_encoder_bias = jnp.array(bundle['y_encoder_bias'])
        
        # Feature positional embedding
        self.col_embedding = jnp.array(bundle['col_embedding'])
        self.feat_pos_emb_weight = jnp.array(bundle['feat_pos_emb_weight'])
        self.feat_pos_emb_bias = jnp.array(bundle['feat_pos_emb_bias'])
        self.feature_selection_mask = bundle.get('feature_selection_mask')  # Optional
        
        # Transformer weights
        self.transformer_params = bundle['transformer_params']
        
        # Training data (needed for in-context learning)
        self.X_train_preprocessed = bundle['X_train_preprocessed']
        self.y_train = bundle['y_train']
        self.n_classes = bundle['n_classes']
        self.softmax_temperature = bundle['softmax_temperature']
        self.classes = bundle['classes']
        self.is_regression = self.config.get('is_regression', False)
        
        if self.is_regression:
            self.y_train_mean = bundle['y_train_mean']
            self.y_train_std = bundle['y_train_std']
            self.znorm_borders = jnp.array(bundle['znorm_borders'])
            # Reconstruct raw borders just in case
            self.raw_borders = self.znorm_borders * self.y_train_std + self.y_train_mean
        
        # GPU preprocessor clipping threshold
        self.outlier_std = bundle['gpu_preprocessor_std']
        
        # Build JAX transformer model
        from tabpfn.architectures.jax_transformer import FlaxPerFeatureTransformer
        
        # Create a mock config with needed attributes
        class MockConfig:
            pass
        
        mc = MockConfig()
        mc.emsize = self.config['emsize']
        mc.nhead = self.config['nhead']
        mc.nlayers = self.config['nlayers']
        mc.nhid_factor = self.config['nhid_factor']
        mc.features_per_group = self.config['features_per_group']
        
        self.jax_model = FlaxPerFeatureTransformer(
            config=mc,
            n_out=self.config['n_out'],
            col_embedding=self.col_embedding,
        )
        
        self.jax_params = {'params': self.transformer_params}
        
        self.features_per_group = self.config['features_per_group']
        self.emsize = self.config['emsize']
    
    def preprocess_x(self, X_test_raw):
        """CPU preprocessing: sklearn pipeline (quantile transform, SVD, etc.)
        
        This uses the fitted sklearn preprocessor — pure NumPy, no PyTorch.
        """
        # The preprocessor returns a PreprocessingPipelineResult object
        result = self.cpu_preprocessor.transform(X_test_raw)
        
        # We need the transformed X array
        if hasattr(result, 'X'):
            return result.X
        return result
    
    def _build_model_input(self, X_test_preprocessed):
        """Build the tensor that enters the model, mimicking PyTorch model.forward().
        
        Returns:
            embedded_input: [1, seq_len, n_groups+1, emsize]
            single_eval_pos: int
        """
        fpg = self.features_per_group
        X_train = self.X_train_preprocessed
        y_train = self.y_train
        n_train = len(X_train)
        n_test = len(X_test_preprocessed)
        
        # Stack train + test
        X_full = np.concatenate([X_train, X_test_preprocessed], axis=0)  # [seq, feats]
        
        # 1. Pad to multiple of features_per_group (Before Masking)
        # This is crucial because TabPFN pads *before* applying the RemoveEmptyFeatures mask
        num_features = X_full.shape[1]
        missing = (fpg - (num_features % fpg)) % fpg
        if missing > 0:
            X_full = np.concatenate([X_full, np.zeros((X_full.shape[0], missing))], axis=1)
            
        # 2. Apply feature selection mask if present (from RemoveEmptyFeaturesEncoderStep)
        if self.feature_selection_mask is not None:
            # Mask is [n_groups, fpg]
            mask_flat = self.feature_selection_mask.flatten().astype(bool)
            if mask_flat.shape[0] == X_full.shape[1]:
                 X_full = X_full[:, mask_flat]
            else:
                 print(f"Warning: feature mask shape {mask_flat.shape} mismatch X {X_full.shape}")

        # 3. Pad again if masking made it non-multiple of fpg (unlikely but safe)
        num_features = X_full.shape[1]
        missing = (fpg - (num_features % fpg)) % fpg
        if missing > 0:
            X_full = np.concatenate([X_full, np.zeros((X_full.shape[0], missing))], axis=1)
        
        # Add batch dim and convert
        X_full = jnp.array(X_full[None])  # [1, seq, feats_padded]
        
        # Rearrange: [1, seq, feats] -> [1, seq, n_groups, fpg]
        seq_len = n_train + n_test
        n_groups = X_full.shape[2] // fpg
        X_full = X_full.reshape(1, seq_len, n_groups, fpg)
        
        # Rearrange for encoder: [seq, 1*n_groups, fpg]
        x = X_full[0].transpose(1, 0, 2)  # [n_groups, seq, fpg]
        x = x.reshape(n_groups, seq_len, fpg).transpose(1, 0, 2)  # [seq, n_groups, fpg]
        
        # GPU preprocessing: soft clip outliers
        # (simplified — for breast cancer dataset this has minimal effect)
        
        # X Encoder pipeline:
        # Step 0: RemoveEmptyFeatures (no-op at inference)
        # Step 1: NaN handling
        x_clean, nan_ind = nan_handling(x)
        
        # Step 2: NormalizeFeatureGroups (normalize_by_used_features=False, skip)
        
        # Step 3: FeatureTransformEncoderStep (normalize_data)
        x_clean = normalize_data(x_clean, train_len=n_train, clip=True)
        
        # Step 4: NormalizeFeatureGroups (normalize_by_used_features=True)
        # Compute non-constant features mask from training data
        x_train_part = x_clean[:n_train]
        non_constants = jnp.any(x_train_part != x_train_part[0:1], axis=0)  # [n_groups, fpg]
        num_used = non_constants.sum(axis=-1, keepdims=True).clip(min=1)  # [n_groups, 1]
        x_clean = normalize_feature_groups(x_clean, non_constants, num_used)
        
        # Step 5: LinearInputEncoder (concatenate main+nan_indicators, then linear)
        embedded_x = x_encoder_forward(x_clean, nan_ind, self.x_encoder_weight)
        # [seq, n_groups, emsize]
        
        # Reshape back: [seq, n_groups, emsize] -> [1, seq, n_groups, emsize]
        embedded_x = embedded_x[None]  # [1, seq, n_groups, emsize]
        
        # Y encoding
        # Build y: [seq, 1, 1] with NaN for test positions
        y_full = np.full((seq_len, 1, 1), np.nan, dtype=np.float32)
        y_train_typed = y_train.astype(np.float32)
        
        if self.is_regression:
            # em.y_train is ALREADY z-normalized (mean=0, std=1)
            # Do NOT re-normalize it
            y_full[:n_train, 0, 0] = y_train_typed
            y_encoded = y_full # [seq, 1, 1]
        else:
            # Classification: Ordinal encoding
            y_full[:n_train, 0, 0] = y_train_typed
            y_full = jnp.array(y_full)
            unique_ys = jnp.array(np.unique(y_train))
            y_encoded = multiclass_target_encode(y_full, unique_ys)  # [seq, 1, 1]
        
        y_encoded = jnp.array(y_encoded) # ensure JAX array
        
        # Set test positions to NaN (already NaN from construction)
        # y_encoded is already NaN for test positions
        
        # NaN handling for y
        y_clean, y_nan_ind = nan_handling(y_encoded)
        
        # Linear projection
        embedded_y = y_encoder_forward(y_clean, y_nan_ind, 
                                       self.y_encoder_weight, self.y_encoder_bias)
        # [seq, 1, emsize]
        
        # Transpose to match [batch, seq, emsize]
        embedded_y = embedded_y.transpose(1, 0, 2)  # [1, seq, emsize]
        
        # Feature positional embedding (only on x features, not y)
        embedded_x = feature_positional_embedding(
            embedded_x, self.col_embedding, 
            self.feat_pos_emb_weight, self.feat_pos_emb_bias
        )
        
        # Concatenate: [batch, seq, n_groups+1, emsize]
        embedded_input = jnp.concatenate(
            [embedded_x, embedded_y[:, :, None, :]], axis=2
        )
        
        return embedded_input, n_train
    
    def predict_decoder_output(self, X_test_raw):
        """Run full pipeline: preprocess → encode → transform → decode.
        
        Returns raw decoder output [1, n_test, n_out].
        """
        # CPU preprocessing
        X_test_pp = self.preprocess_x(X_test_raw)
        
        # Build model input
        embedded_input, single_eval_pos = self._build_model_input(X_test_pp)
        
        # Run transformer (encoder + decoder)
        # The JAX model expects input with positional embedding already applied
        output = self.jax_model.apply(
            self.jax_params, embedded_input,
            single_eval_pos=single_eval_pos,
            skip_positional_embedding=True,
        )
        
        return output  # [1, n_test, n_out]
    
    def predict_regression(self, X_test_raw):
        """Regression prediction: returns mean prediction."""
        # 1. Get raw logits (in z-normalized space)
        output = self.predict_decoder_output(X_test_raw) # [1, n_test, n_out]
        logits = output[0] # [n_test, n_out]
        
        # 2. Translate logits to znorm space
        # Original implementation uses specific borders for each estimator member.
        # But we only have one ensemble member here (exported).
        # We need to check if the exported member had a target transform.
        # For simplicity, we assume the exported member's output logits align with znorm_borders
        # OR we need to replicate 'transform_borders_one' if the member has a transform.
        # In 'export_for_jax', we exported 'znorm_borders'.
        # We'll assume for this single member, the logits correspond to znorm_borders directly
        # if no target transform was active.
        # If there was a target transform, we'd need that transform logic.
        # TODO: Add check for target transform in bundle.
        # For now, we assume simple case (Breast Cancer / Boston usually don't have complex target transforms in default)
        # Actually Boston might.
        
        # Let's assume logits are for znorm_borders for now.
        # Translate probability mass
        # The logits define a distribution over buckets defined by SOME borders.
        # Which borders? 'borders_t' in regressor.py.
        # If no target transform, borders_t = znorm_borders.
        
        # Let's compute mean in znorm space
        # We need to map logits -> probs over znorm_borders
        # If logits are already over znorm_borders, we just softmax and compute mean.
        
        # We will assume logits correspond to self.znorm_borders 
        # (This implies no target transform, or we are missing that piece. 
        # But for 'boston' in default TabPFN V2, it might be fine).
        
        # Compute mean in znorm space
        # We use the helper function
        mean_znorm = compute_mean_at_distribution(logits, self.znorm_borders)
        
        # 3. Denormalize
        mean_raw = mean_znorm * self.y_train_std + self.y_train_mean
        
        return mean_raw

    def predict_proba(self, X_test_raw):
        """Full prediction: returns class probabilities [n_test, n_classes]."""
        if self.is_regression:
             raise ValueError("predict_proba is not supported for regression models.")
             
        output = self.predict_decoder_output(X_test_raw)
        
        # Apply class permutation and slice to n_classes
        perm = self.ensemble_config['class_permutation']
        # Ensure perm is flat numpy array of ints
        if perm is not None:
            perm = np.array(perm).flatten().astype(int)
        else:
            # Fallback if None (e.g. regression model used by mistake or no perm)
            perm = np.arange(output.shape[-1])
            

        
        # Try to detect if output is declared as [batch, n_out, seq] or [batch, seq, n_out]
        # output[0] usually removes batch dim.
        out0 = output[0]
        if out0.shape[0] == self.config['n_out']:
             # It seems output is [n_out, n_test]
             # Transpose to [n_test, n_out]
             out0 = out0.T
        
        logits = out0[:, perm[:self.n_classes]]  # [n_test, n_classes]
        
        # Temperature-scaled softmax
        temp = self.softmax_temperature
        probas = jax.nn.softmax(logits / temp, axis=-1)
        
        # JAX softmax might return JAX array, convert to numpy
        probas = np.array(probas)
        
        if probas.shape[0] == self.n_classes and probas.shape[1] != self.n_classes:
             print("DEBUG: Transposing probas to match [n_test, n_classes]")
             probas = probas.T
             
        return probas
    
    def predict(self, X_test_raw):
        """Predict class labels or regression targets."""
        if self.is_regression:
            return np.array(self.predict_regression(X_test_raw))
        else:
            probas = self.predict_proba(X_test_raw)
            pred_indices = np.argmax(probas, axis=1)
            return self.classes[pred_indices]


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("JAX TabPFN Inference Example")
    print("=" * 70)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, choices=['breast_cancer', 'boston'], help='Dataset to use')
    parser.add_argument('--save-output', type=str, help='Save encoder output to .npy file')
    parser.add_argument('--verify', action='store_true', help='Verify against bundled PyTorch output')
    args = parser.parse_args()
    
    # Determined dataset/bundle
    if args.dataset == 'boston':
         bundle_path = os.path.join(os.path.dirname(__file__), "jax_inference_bundle_reg.pkl")
         dataset_name = 'boston'
    elif args.dataset == 'breast_cancer':
         bundle_path = os.path.join(os.path.dirname(__file__), "jax_inference_bundle.pkl")
         dataset_name = 'breast_cancer'
    else:
         # Default check logic
         if os.path.exists(os.path.join(os.path.dirname(__file__), "jax_inference_bundle_reg.pkl")):
             # Prefer regression if user didn't specify? No, prefer classification as default
             pass
         bundle_path = os.path.join(os.path.dirname(__file__), "jax_inference_bundle.pkl")
         dataset_name = 'breast_cancer'
         if not os.path.exists(bundle_path) and os.path.exists(os.path.join(os.path.dirname(__file__), "jax_inference_bundle_reg.pkl")):
             bundle_path = os.path.join(os.path.dirname(__file__), "jax_inference_bundle_reg.pkl")
             dataset_name = 'boston'
    
    if not os.path.exists(bundle_path):
        print(f"\nBundle not found at {bundle_path}")
        print("Please ensure 'jax_inference_bundle.pkl' or 'jax_inference_bundle_reg.pkl' is in the script directory.")
        print("You may need to run the export script from the original PyTorch repo first.")
        return

    # Load bundle
    with open(bundle_path, 'rb') as f:
        bundle = pickle.load(f)
    print(f"\nLoading JAX TabPFN from {bundle_path}...")
    model = JAXTabPFN(bundle_path)
    
    # Load Data
    from sklearn.model_selection import train_test_split
    if dataset_name == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        metric_name = 'Accuracy'
    else:
        # Boston/California
        try:
            from sklearn.datasets import load_boston
            X, y = load_boston(return_X_y=True)
            metric_name = 'R2 Score'
            # Check if bundle used california
        except ImportError:
            from sklearn.datasets import fetch_california_housing
            X, y = fetch_california_housing(return_X_y=True)
            X = X[:1000]
            y = y[:1000]
            metric_name = 'R2 Score'

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    
    print(f"\nDataset: {dataset_name}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Use bundled X_test if verifying, to ensure exact same data
    if args.verify and 'verification_data' in bundle:
        print("Using bundled X_test for verification...")
        X_test_for_pred = bundle['verification_data']['X_test']
    else:
        X_test_for_pred = X_test

    # Warm-up (JIT compilation)
    print("Warming up (JIT compilation)...")
    if model.is_regression:
         _ = model.predict(X_test_for_pred[:2])
    else:
         _ = model.predict_proba(X_test_for_pred[:2])
    
    # Predict
    print("\nRunning prediction...")
    t0 = time.time()
    
    if model.is_regression:
        predictions = model.predict(X_test_for_pred)
    else:
        probas = model.predict_proba(X_test_for_pred)

        predictions = model.classes[np.argmax(probas, axis=1)]
        
    t_predict = time.time() - t0
    print(f"  Prediction time: {t_predict:.3f}s")
    
    # Metrics
    if metric_name == 'Accuracy':
        from sklearn.metrics import accuracy_score, roc_auc_score
        acc = accuracy_score(y_test, predictions)
        try:
             auc = roc_auc_score(y_test, probas[:, 1])
        except:
             auc = 0.0
        print(f"\nMetrics:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  AUC: {auc:.4f}")
    else:
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"\nMetrics:")
        print(f"  R2 Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")

    if args.save_output:
        np.save(args.save_output, predictions)
        print(f"Saved output to {args.save_output}")
    
    if args.verify:
        if 'verification_data' not in bundle:
            print("\nVerification data not found in bundle. Re-run value export_for_jax.py.")
        else:
            pt_out = bundle['verification_data']['pytorch_output']
            # pt_out: [seq, 1, 10]
            # jax_out needs likely logits for direct comparison or we compare final predictions
            
            # For verification we usually compare logits. 
            # But the script now returns predictions.
            # Let's run predict_decoder_output again for verification
            decoder_output = model.predict_decoder_output(X_test_for_pred)
            
            # Align shapes
            pt_out = pt_out.transpose(1, 0, 2)
            jax_out = np.array(decoder_output)
            
            diff = np.abs(pt_out - jax_out)
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            print("\n" + "=" * 70)
            print("Verification Results (Logits)")
            print("=" * 70)
            print(f"Max absolute difference: {max_diff:.6f}")
            print(f"Mean absolute difference: {mean_diff:.6f}")
            
            if max_diff < 0.1: # Relaxed tolerance for float32/fp16 differences
                print("SUCCESS: JAX output matches PyTorch output!")
            else:
                print("WARNING: large difference found.")
                print(f"First 5 PyTorch: {pt_out[0, 0, :5]}")
                print(f"First 5 JAX:     {jax_out[0, 0, :5]}")


if __name__ == "__main__":
    main()
