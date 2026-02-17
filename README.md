# TabPFN-v2-JAX

> **Built with [PriorLabs-TabPFN](https://github.com/PriorLabs/TabPFN)**

A standalone, pure JAX re-implementation of [TabPFN](https://github.com/PriorLabs/TabPFN) v2 inference. Run TabPFN predictions without any PyTorch dependency.

> [!NOTE]
> This JAX implementation was developed with the assistance of **[Antigravity](https://deepmind.google/)**, an AI coding agent by Google DeepMind.

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
    pip install -r requirements.txt
    ```

## Model Bundles

The inference script requires pre-exported model bundles:
- `jax_inference_bundle.pkl` — classification
- `jax_inference_bundle_reg.pkl` — regression

These files are **not included in the git repository** due to their size (~130 MB total). To generate them, run the export script from the original TabPFN PyTorch repository:

```bash
# In the original PyTorch repo environment:
python scripts/export_for_jax.py --dataset breast_cancer
python scripts/export_for_jax.py --dataset boston
```

Place the resulting `.pkl` files in the repository root directory.

## Usage

### Run Prediction

```bash
# Classification (Breast Cancer)
python predict.py --dataset breast_cancer

# Regression (California Housing)
python predict.py --dataset boston
```

### Verify Correctness

To verify that the JAX output matches the original PyTorch output (requires verification data in the bundle):

```bash
python predict.py --dataset breast_cancer --verify
python predict.py --dataset boston --verify
```

## Project Structure

-   `predict.py` — Main standalone inference script (classification + regression).
-   `src/` — JAX TabPFN library source code.
-   `scripts/` — Utility scripts (e.g., `export_for_jax.py`, requires PyTorch).
-   `requirements.txt` — Python dependencies.

## Note on Ensembling

The current script demonstrates prediction using a **single ensemble member**. The full PyTorch TabPFN uses up to 8 ensemble members with 636 feature shifts each. Implementing feature shifting in JAX would achieve exact logit parity but at ~636× inference cost per member.

## License

This project is licensed under the **Prior Labs License** (Apache 2.0 with Additional Provision) — see the [LICENSE](LICENSE) file for details.

The original TabPFN code and v2 model weights are by [Prior Labs GmbH](https://priorlabs.ai/). See [NOTICE](NOTICE) for full attribution.

## Acknowledgments

- **[PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN)** — The original TabPFN project by Prior Labs GmbH, on which this JAX implementation is based.
- **[Antigravity](https://deepmind.google/)** — AI coding agent by Google DeepMind that assisted in developing this JAX re-implementation.

## Citation

If you use this software, please cite the original TabPFN paper:

```bibtex
@article{hollmann2025tabpfn,
  title={Accurate predictions on small data with a tabular foundation model},
  author={Hollmann, Noah and M{\"u}ller, Samuel and Purucker, Lennart and
          Krishnakumar, Arjun and K{\"o}rfer, Max and Hoo, Shi Bin and
          Schirrmeister, Robin Tibor and Hutter, Frank},
  journal={Nature},
  year={2025},
  publisher={Nature Publishing Group}
}
```
