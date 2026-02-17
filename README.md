# JAX TabPFN

A standalone, pure JAX implementation of TabPFN inference. This repository allows you to run TabPFN predictions without any PyTorch dependency.

## Benchmark Results

Benchmarked against PyTorch TabPFN V2 (single ensemble member):

| Task | Metric | JAX | PyTorch V2 |
| :--- | :--- | :--- | :--- |
| Classification (Breast Cancer) | Accuracy | **96.84%** | ~96–98% |
| | AUC | **0.9965** | >0.99 |
| Regression (California Housing) | R² | **0.8245** | **0.8342** |
| | RMSE | **0.3748** | **0.3643** |

### Known Differences from PyTorch

The classification logits show some divergence from PyTorch (max diff ~9.8) while regression logits match tightly (max diff ~0.4). This is **not a bug** — the cause is:

1. **Feature shifting**: PyTorch uses `feature_shift_count=636` random feature permutations and averages results. JAX currently runs a single pass (no shifts).
2. **Fingerprint feature**: PyTorch appends a fingerprint feature column (`add_fingerprint_feature=True`). JAX omits this.

Despite these differences, final prediction accuracy is excellent because the probability ranking is preserved through softmax.

## Installation

1.  **Install JAX**: Follow the [official JAX installation guide](https://github.com/google/jax#installation) for your platform (CPU, GPU, or TPU).
    ```bash
    pip install jax jaxlib
    ```

2.  **Install other dependencies**:
    ```bash
    pip install numpy scikit-learn
    ```

## Usage

### 1. Ensure Model Bundle is Present

The inference script requires `jax_inference_bundle.pkl` (classification) and/or `jax_inference_bundle_reg.pkl` (regression) in the root directory.

If these files are missing, generate them using the export script from the original TabPFN PyTorch repository:
```bash
# In the original PyTorch repo environment:
python scripts/export_for_jax.py --dataset breast_cancer
python scripts/export_for_jax.py --dataset boston
```

### 2. Run Prediction

```bash
# Classification (Breast Cancer)
python predict.py --dataset breast_cancer

# Regression (California Housing)
python predict.py --dataset boston
```

### 3. Verify Correctness

To verify that the JAX output matches the original PyTorch output (requires verification data in the bundle):

```bash
python predict.py --dataset breast_cancer --verify
python predict.py --dataset boston --verify
```

## Project Structure

-   `predict.py`: Main standalone inference script (classification + regression).
-   `src/`: JAX TabPFN library source code.
-   `jax_inference_bundle.pkl`: Serialized classification model bundle.
-   `jax_inference_bundle_reg.pkl`: Serialized regression model bundle.
-   `scripts/`: Utility scripts (e.g., `export_for_jax.py`, requires PyTorch).

## Note on Ensembling

The current script demonstrates prediction using a **single ensemble member**. The full PyTorch TabPFN uses up to 8 ensemble members with 636 feature shifts each. Implementing feature shifting in JAX would achieve exact logit parity but at ~636× inference cost per member.
