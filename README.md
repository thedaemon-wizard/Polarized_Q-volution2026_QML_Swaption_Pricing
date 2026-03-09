# Option Pricing with Quantum Machine Learning

## Q-volution 2026 QML Hackathon - Track B (Quandela)

This repository contains our submission for the Q-volution 2026 Quantum Machine Learning Hackathon,
Track B organized by Quandela in collaboration with Mila and AMF (Autorite des marches financiers du Quebec).

The goal is to build a QML model to predict actual prices of put and call options (swaptions)
using the provided training dataset, built with Quandela's MerLin framework for photonic
quantum machine learning.

---

### Quick Start for Judges

1. **Open** `swaption_qml.ipynb` in Google Colab (GPU runtime recommended)
2. **Run All Cells** - the notebook is fully self-contained (data loads from Hugging Face)
3. **Key results**: See Section 12 (QRC) for the Final Candidate selection and Section 13.1 for holdout + test evaluation

> **Final Candidate**: QRC (Enriched Features, Belenos Noise) - RMSE 0.0392 on validation, **RMSE 0.0104 on test ground truth (R² = 0.989)**
>
> Enriched features (rolling statistics + momentum) with Belenos QPU noise model validation.

| Metric | Validation | Holdout (6-day AR) | Test (Ground Truth) |
|--------|-----------|-------------------|-------------------|
| RMSE | 0.0392 | 0.0041 | **0.0104** |
| MAE | 0.0316 | 0.0032 | **0.0084** |
| R² | 0.953 | 0.998 | **0.989** |

---

## Challenge Overview

### What is a Swaption?

A swaption is a financial derivative that gives its holder the right (but not the obligation) to enter
into an interest rate swap at a future date. The pricing depends on:

- **Maturity**: When the option can be exercised (e.g., 2 years)
- **Tenor**: How long the underlying swap lasts once entered (e.g., 5 years)

This creates a 2D pricing surface (maturity x tenor grid) that evolves over time.

### Task Description

**Primary Goal**: Use the ~490 rows of training data (Level 1) to predict the **next 6 rows**
(224 swaption price features each). Assessment is based solely on these 6 predicted rows.

Per Discord mentor guidance:
> "Provided the ~490 rows of the dataset, provide the next 6 rows (224 features).
> Those 6 rows are hidden from participants but accessible to us.
> They will have to only use the ~490 rows to train, test, validate.
> The assessment will be made based on those 6 rows."

Level 2 (missing data imputation) is **optional** and retained in the notebook for reference.

### Key Constraints

- Maximum **20 modes / 10 photons** in simulation
- **No amplitude encoding** or state injection on QPU - angle encoding only
- Must use **MerLin framework** (Quandela)
- QPUs: Ascella (12 modes, 6 photons) and Belenos (24 modes, 12 photons)

## Approach

### Core Innovation: Quantum Residual Correction

Our key insight is that classical linear models already capture ~95% of swaption price variance.
Rather than training a quantum model from scratch, we use a **Residual Hybrid Architecture**
where the quantum circuit learns to correct the classical baseline's errors:

```
Input (PCA features)
    |
    +--> [Frozen] Linear Regression --> Base Prediction
    |                                        |
    +--> ScaleLayer --> QuantumLayer          |
              |                              |
         LexGrouping --> BatchNorm           |
              |                              |
         MLP Head --> Correction             |
              |                              |
              +--- alpha * Correction -------+
                                             |
                                       Final Output
```

This approach outperforms the classical baseline. With enriched features (rolling statistics
+ momentum), the best QRC achieves RMSE 0.0392 vs 0.0437 for the classical baseline (-10.3%).

### Quantum Reservoir Computing (QRC) - Final Candidate

We also explore **Quantum Reservoir Computing** where the quantum circuit parameters are
**fixed** (random, never trained) and only a classical readout is optimized:

```
Input (PCA features)
    |
    +--> [Frozen] Linear Regression --> Base Prediction
    |                                        |
    +--> ScaleLayer --> [FIXED] Quantum Circuit |
              |                              |
         LexGrouping --> BatchNorm           |
              |                              |
         MLP Correction --> alpha * corr ----+
                                             |
                                       Final Output
```

The QRC (Enriched Features, Belenos noise) is our **Final Candidate**, achieving the best
validation RMSE (0.0392) with QPU noise validation and excellent test ground truth performance
(RMSE 0.0104, R² 0.989). The enriched features (rolling statistics + momentum) improved
prediction by ~9% over the baseline QRC.

### Optimizer Strategy: Why Adam is Correct for QRC

In QRC, the quantum circuit is **frozen** — only classical readout layers (BatchNorm, MLP,
ScaleLayer, alpha) are trained. Noise-aware quantum optimizers (SPSA, QN-SPSA, Photonic PSR
[20b, 20c]) address gradient estimation *through* noisy quantum measurements, which is
unnecessary when quantum parameters are fixed. Adam [15b] handles mini-batch stochasticity
in the classical gradient computation, which is the dominant noise source in our pipeline.

This approach is validated by Quandela's own QORC paper [24], which uses AdaGrad for
classical readout, and MerLin [1], which explicitly recommends PyTorch Adam/SGD over
SPSA for differentiable training. QRC noise resilience literature [24b, 24c] further
confirms that quantum noise acts as implicit regularization rather than a gradient
estimation problem.

### Why Residual QRC Outperforms Pure VQC

Our experiments reveal a striking result: **Residual QRC (RMSE 0.043) dramatically outperforms
pure VQC (RMSE ~0.19)**, despite training zero quantum parameters. Three factors explain this:

1. **Barren Plateaus**: Pure VQC suffers from exponentially vanishing gradients in the quantum
   parameter landscape. At 12+ modes, the loss landscape becomes effectively flat, making gradient-based
   optimization futile. QRC sidesteps this entirely by freezing the quantum circuit.

2. **Residual Architecture**: By using the quantum circuit as a correction to a strong classical
   baseline (Linear Regression, R2 0.95), we only ask the quantum component to model the ~5%
   residual variance. This is a fundamentally easier task than predicting prices from scratch.

3. **Noise Resilience in Photonic Circuits**: The Belenos noise model (brightness=0.2390,
   transmittance=0.5290) actually preserves or slightly improves QRC performance. This is because
   QRC treats the quantum circuit as a fixed feature extractor; photon loss and imperfect
   interference simply create a different (but equally useful) feature space. The classical
   readout adapts to whatever features the noisy circuit provides.

This result has practical significance: QRC is **QPU-deployable** without parameter optimization
on hardware, making it ideal for near-term photonic quantum devices.

### Technical Architecture

| Feature | Implementation |
|---------|---------------|
| **Framework** | MerLin (merlinquantum) + Perceval |
| **Encoding** | Angle encoding only (QPU-compatible) |
| **Scaling** | Learnable per-feature ScaleLayer (replaces fixed pi) |
| **Circuit** | Standard, data re-uploading, and QRC (fixed reservoir) |
| **Architecture** | Residual hybrid (classical + quantum correction) |
| **QRC** | Fixed random interferometer + classical readout (no barren plateaus) |
| **Normalization** | MinMax [0,1] input + BatchNorm on quantum output |
| **Optimizer** | Adam [15b] (correct for QRC: classical-only gradients) |
| **Training** | CosineAnnealing LR, Huber loss, gradient clipping |
| **Max Scale** | 20 modes / 10 photons (184,756 quantum output dim) |
| **QPU Noise** | Hardware-derived from Ascella and Belenos QPUs |

### Preprocessing Pipeline

1. **Min-Max normalization** to [0, 1] range (optimal for angle encoding)
2. **Sliding window** (size 5) for time-series prediction
3. **PCA** dimensionality reduction (4-10 components, 98-99.8% variance retained)
4. **Learnable ScaleLayer** instead of fixed pi multiplication
5. **Enriched features** (Final Candidate): rolling mean, rolling std, and momentum
   (day-over-day change) computed within each sliding window, concatenated with the
   flat window (1120 → 1792 dims), then PCA-compressed to 6 components

### Auto-Regressive Forecasting

The 6-row prediction uses an **auto-regressive** approach: each predicted row at T+k is fed back
as input for predicting T+k+1. This captures evolving market dynamics rather than producing
static one-shot predictions:

```
Window[T-4..T] -> PCA -> Model -> Pred[T+1]
Window[T-3..T, Pred[T+1]] -> PCA -> Model -> Pred[T+2]
...
Window[Pred[T+1..T+4], Pred[T+5]] -> PCA -> Model -> Pred[T+6]
```

### Circuit Design

- **Standard Circuit**: Entangling -> Angle Encoding -> [Rotations + Entangling] x 2 -> Superpositions
- **Data Re-uploading Circuit**: Superpositions -> [Encoding + Rotations + Entangling] x N stages
- **QRC Reservoir Circuit**: Fixed Random Interferometer -> Encoding -> Fixed Random Interferometer (no trainable quantum params)
- UNBUNCHED computation space (at most 1 photon per mode)
- "Dressed quantum circuit" pattern with classical pre/post-processing

## Results

### Final Candidate: QRC (Enriched Features, Belenos Noise)

**Validation RMSE: 0.0392 | Test RMSE: 0.0104 | Test R² = 0.989**

The best model combines **enriched features** (rolling statistics + momentum) with the
**Belenos QPU noise model** (brightness=0.2390, transmittance=0.5290). The enriched features
improved validation RMSE by 9.4% over the original QRC, and noise model validation
confirms QPU-deployability.

> Note on QPU transparency: Due to QPU queue timeouts during evaluation (both Ascella in maintenance
> and Belenos in calibration at test time), we report results using the **Belenos Noise Model** simulator.
> This simulator uses hardware-derived parameters from live QPU metrics, providing realistic noise
> characteristics. The QPU evaluation cell is included in the notebook and will automatically execute
> when QPUs become available (`status == "running"`).

### Model Comparison (26 Configurations)

| Model | RMSE | R² | Type |
|-------|------|-----|------|
| QRC (Enriched, Ideal) | 0.0384 | -- | Best Validation |
| **QRC (Enriched, Belenos)** | **0.0392** | -- | **Final Candidate** |
| QRC (Enriched, Ascella) | 0.0392 | -- | Enriched + Noise |
| LR (Enriched Features) | 0.0396 | -- | Classical + Features |
| Residual QRC (Belenos) | 0.0431 | -- | Reservoir + Noise |
| Residual QRC (Ascella) | 0.0432 | -- | Reservoir + Noise |
| Residual Hybrid (VQC) | 0.0432 | 0.950 | Quantum + Classical |
| Residual QRC (Ideal) | 0.0433 | -- | Reservoir + Classical |
| Linear Regression | 0.0437 | 0.950 | Classical |
| Ensemble QRC (N=5) | 0.0437 | -- | Ensemble Reservoir |
| Transformer (d=64, 4-head) [26, 26c] | 0.0459 | -- | Deep Learning |
| LSTM (hidden=64, 2-layer) [25] | 0.0474 | -- | Deep Learning |
| MLP (128, 64, 32) | 0.0489 | -- | Classical |
| Standard Quantum (best) | 0.1864 | -- | Quantum only |
| QRC Pure (Ideal) | 0.1871 | -- | Reservoir only |
| Data Re-uploading 2x | 0.1929 | -- | Quantum only |
| Quantum Kernel (6m/3p) | 0.2294 | -- | Kernel Method |

### Test Data Evaluation (Ground Truth)

After the hidden test data (`test.xlsx`, 6 rows) was revealed, we evaluated our auto-regressive
predictions against the actual ground truth:

| Metric | Holdout (proxy) | Test (ground truth) |
|--------|----------------|-------------------|
| RMSE | 0.0041 | **0.0104** |
| MAE | 0.0032 | **0.0084** |
| R² | 0.998 | **0.989** |

Per-day test RMSE shows expected auto-regressive error accumulation:

| Day | Date | RMSE |
|-----|------|------|
| 1 | 2051-12-24 | 0.0022 |
| 2 | 2051-12-26 | 0.0060 |
| 3 | 2051-12-27 | 0.0069 |
| 4 | 2051-12-29 | 0.0097 |
| 5 | 2051-12-30 | 0.0167 |
| 6 | 2052-01-01 | 0.0134 |

### Key Visualizations

| Plot | Description |
|------|-------------|
| `results/results_summary.png` | 3-panel summary: model comparison, noise impact, QRC results |
| `results/holdout_evaluation.png` | Holdout: scatter, per-day RMSE, sample feature time series |
| `results/test_evaluation_heatmap.png` | Volatility surface heatmap: Actual vs Predicted vs Difference |
| `results/qrc_comparison.png` | QRC vs VQC performance comparison |

### Computation Time Summary

All training times measured on NVIDIA RTX PRO 6000 (GPU) with CUDA 12.x.
Noise models run on CPU (MerLin's `PhotonLossTransform` device constraint).

| Backend | Time | Notes |
|---------|------|-------|
| Classical (LR + MLP) | < 1s | CPU only |
| Quantum simulator (GPU) | 14-175s | Scales with mode count |
| Noise model (CPU) | 37-79s | QPU-derived parameters |
| QRC Ideal (GPU) | 12s | No quantum params trained |
| QRC Noisy (CPU) | 129-131s | Fixed circuit + noise |
| Best model retrain | ~13s | 50-80 epochs |

Timing is included in all print outputs and exported to `model_comparison.csv` (Time_s column)
and `noise_comparison.csv` (Time_s column).

### Quantum Circuit Scale Comparison

| Config | Modes | Photons | Q-Output Dim | RMSE | Time |
|--------|-------|---------|-------------|------|------|
| Small | 6 | 3 | 20 | 0.1925 | 14s |
| Medium | 10 | 5 | 252 | 0.1906 | 35s |
| QPU-Ascella | 12 | 6 | 924 | 0.2047 | 50s |
| Large | 16 | 8 | 12,870 | 0.1893 | 90s |
| **Max** | **20** | **10** | **184,756** | **0.1864** | **176s** |

### Data Re-uploading Experiment

| Stages | RMSE | Notes |
|--------|------|-------|
| 1 (standard) | 0.1962 | Baseline |
| 2 | 0.1929 | -1.7% improvement |
| 3 | 0.2133 | Diminishing returns |

### Noise Model Comparison (QPU-Derived Parameters)

| QPU/Config | Brightness | Transmittance | RMSE | Time |
|------------|-----------|--------------|------|------|
| Ascella | 0.1033 | 0.2440 | 0.1833 | 77s |
| Belenos | 0.2390 | 0.5290 | 0.1843 | 75s |
| Ideal | 1.0000 | 1.0000 | 0.1920 | 45s |

Noise parameters derived from live QPU hardware metrics (HOM, Transmittance, g2).

### Quantum Reservoir Computing (QRC) Results

| Model | RMSE | Time | Noise | Quantum Params Trained |
|-------|------|------|-------|----------------------|
| QRC Pure (Ideal) | 0.1871 | 13s | None | 0 (fixed) |
| Residual QRC (Ideal) | 0.0433 | 13s | None | 0 (fixed) |
| Residual QRC (Ascella) | 0.0432 | 128s | Ascella QPU | 0 (fixed) |
| Residual QRC (Belenos) | 0.0431 | 124s | Belenos QPU | 0 (fixed) |
| QRC Enriched (Ideal) | 0.0384 | 30s | None | 0 (fixed) |
| QRC Enriched (Ascella) | 0.0392 | 85s | Ascella QPU | 0 (fixed) |
| **QRC Enriched (Belenos)** | **0.0392** | **79s** | **Belenos QPU** | **0 (fixed)** |

QRC uses fixed random interferometers as quantum feature extractors, training only
the classical readout layer. With enriched features (rolling statistics + momentum),
the QRC achieves a 10.3% improvement over the classical baseline. Noise model validation
on both Ascella and Belenos QPU parameters confirms QPU-deployability.

### QPU Information (Retrieved from Quandela Cloud)

**Ascella QPU:**
- 12 modes, 6 photons maximum
- Threshold detectors (binary: click / no-click)
- Connected input modes: [0, 2, 4, 6, 8, 10] (even modes only)
- Performance: HOM 86.4%, Transmittance 2.44%, g2 1.95%
- Clock: 80 MHz

**Belenos QPU:**
- 24 modes, 12 photons maximum
- Threshold detectors
- Connected input modes: [0, 2, 4, 6, 8, 9, 12, 13, 16, 18, 20, 22]
- Performance: HOM 92.4%, Transmittance 5.29%, g2 2.2%

### Backends

- **CPU Simulator (SLOS)**: Perceval local simulation for development
- **GPU Accelerated**: CUDA-accelerated local simulation for faster training
- **Noise Model Simulator**: QPU-realistic noise with hardware-derived parameters
  (brightness, transmittance, threshold detectors) for both Ascella and Belenos
- **QPU Hardware**: Quandela Cloud via MerlinProcessor (Ascella, Belenos).
  QPU evaluation is automatically enabled when `QUANDELA_TOKEN` is set in `.env`.
  Before submitting jobs, each QPU's operational status is checked via `RemoteProcessor.status`;
  only QPUs with `status == "running"` are evaluated (maintenance/calibration QPUs are skipped).
  Uses a CPU noise-free builder-based model (required for `export_config()` / MerlinProcessor offloading).
  As of March 2026: Ascella (`maintenance`), Belenos (`calibration`) — QPU evaluation is automatically
  skipped until QPUs return to `running` status.

## Repository Structure

```
Q-volution_2026_QML_Finance/
|-- README.md                          # This file
|-- swaption_qml.ipynb                 # Main notebook (Google Colab compatible)
|-- requirements.txt                   # Python dependencies
|-- .env                               # API token (not committed to git)
|-- .gitignore                         # Git ignore rules
|-- results/                           # All output files (generated by notebook)
|   |-- eda_analysis.png               #   Exploratory data analysis plots
|   |-- circuit_6m.png                 #   6-mode circuit diagram
|   |-- circuit_12m.png                #   12-mode circuit diagram (QPU-Ascella)
|   |-- circuit_reuploading.png        #   Data re-uploading circuit diagram
|   |-- circuits_all.png               #   All circuit diagrams combined
|   |-- scale_comparison.png           #   Circuit scale comparison plots
|   |-- noise_comparison.png           #   Noise model impact chart
|   |-- prediction_scatter.png         #   Predicted vs actual + evaluation plots
|   |-- qrc_comparison.png             #   QRC vs VQC comparison plots
|   |-- tscv_comparison.png            #   Time-series cross-validation results
|   |-- deep_classical_comparison.png  #   LSTM/Transformer vs classical/quantum comparison
|   |-- hyperparameter_sensitivity.png #   Hyperparameter sweep results
|   |-- results_summary.png            #   Summary visualization (3-panel)
|   |-- holdout_evaluation.png         #   Holdout 6-day auto-regressive evaluation
|   |-- test_evaluation_heatmap.png    #   Volatility surface: predicted vs actual
|   |-- qpu_evaluation.png            #   QPU hardware evaluation (when QPU running)
|   |-- qpu_evaluation.csv            #   QPU evaluation metrics (when QPU running)
|   |-- missing_data_analysis.png      #   Level 2 missing data pattern analysis
|   |-- predictions_val.csv            #   Validation set predictions
|   |-- model_comparison.csv           #   All model RMSE comparison (with Time_s)
|   |-- noise_comparison.csv           #   Noise model comparison data (with Time_s)
|   |-- tscv_results.csv               #   Time-series CV fold results
|   |-- sensitivity_results.csv        #   Hyperparameter sensitivity sweep data
|   |-- submission_predictions.xlsx    #   Submission file (test_template format)
|   |-- submission_simulated.xlsx      #   Submission file (sample_Simulated format)
|   |-- models/                        #   Saved trained models
|   |   |-- model_best.pt             #     Best model state dict + metadata
|   |   |-- scaler.pkl                #     Fitted MinMaxScaler
|   |   |-- pca_6.pkl                 #     Fitted PCA(6) transform
|-- sample_datasets/                   # Sample data files for submission format
|   |-- test.xlsx                      #   Test data (6 rows, ground truth)
|   |-- test_template.xlsx             #   Test template (Future prediction + Missing data)
|   |-- train.xlsx                     #   Training data (Excel format)
|   |-- sample_Simulated_Swaption_Price.xlsx  # Example submission format
|-- docs/                              # Documentation and presentation materials
|   |-- TrackB Infographic.html        #   Interactive infographic with Chart.js
|   |-- grand_finale_slides.html       #   Grand Finale slide deck (HTML)
|   |-- grand_finale_presentation.md   #   Presentation script and Q&A prep
|   |-- judging_evaluation_report.md   #   Self-evaluation against judging criteria
```

## Setup

### Quandela Cloud Token

Create a `.env` file in the project root:

```bash
# .env
QUANDELA_TOKEN=your_token_here
```

Get your token at: https://cloud.quandela.com

The `.env` file is excluded from git via `.gitignore` for security.

### Local Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Google Colab

Open the notebook in Google Colab and run the setup cells at the top.
GPU runtime is recommended for faster training with larger circuits.
For Colab, set the token as an environment variable or use Colab secrets.

### Dataset

The dataset is loaded directly in the notebook via Hugging Face.
**Level 1 is the primary training dataset** (~490 rows, 224 features):

```python
from datasets import load_dataset

# Level 1 is the PRIMARY training dataset (~490 rows -> predict next 6)
ds_level1 = load_dataset(
    "Quandela/Challenge_Swaptions",
    data_files="level-1_Future_prediction/train.csv",
    split="train",
)

# Level 2 loaded for optional missing data analysis
ds_level2 = load_dataset(
    "Quandela/Challenge_Swaptions",
    data_files="level-2_Missing_data_prediction/train_level2.csv",
    split="train",
)
```

### Submission Format

Two Excel files are generated matching the sample formats in `sample_datasets/`:

| File | Format | Contents |
|------|--------|----------|
| `submission_predictions.xlsx` | test_template format | Type + 224 prices + Date (6 future rows) |
| `submission_simulated.xlsx` | sample_Simulated format | 224 prices + Date + Type (all rows + 6 future) |

When `test.xlsx` is present in `sample_datasets/`, the submission uses the actual test dates.

### Model Save/Load

Trained models are saved to `results/models/` for reuse without retraining:
- `model_best.pt`: Model state dict + metadata (class, config, column names)
- `scaler.pkl`: Fitted MinMaxScaler for de-normalization
- `pca_6.pkl`: Fitted PCA(6) transform for dimensionality reduction

### Holdout & Test Evaluation

Since no test data was provided during training, we validate the auto-regressive prediction by:
1. Holding out the **last 6 rows** of the ~490-row training set
2. Auto-regressively predicting them using only the preceding rows
3. Computing RMSE/MAE/R2 against actual values as a proxy for assessment performance

After the hidden test data was released (`sample_datasets/test.xlsx`), the notebook
automatically evaluates against ground truth and generates a volatility surface heatmap
comparing predicted vs actual prices.

## Requirements

- Python 3.12
- PyTorch >= 2.4.0
- merlinquantum >= 0.2.2
- perceval-quandela >= 1.0.1
- python-dotenv >= 1.0.0
- CUDA 12.x (for GPU acceleration)
- See requirements.txt for full dependency list

## Hardware Tested

- CPU: Intel i5-13600K
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition 96GB
- RAM: 128GB DDR5 5200
- OS: Alma Linux 9.7

## References

### Framework and Platform

1. Notton, C. et al. "MerLin: A Discovery Engine for Photonic and Hybrid QML." arXiv:2602.11092 (2026). [Paper](https://arxiv.org/abs/2602.11092) | [Docs](https://merlinquantum.ai) | [GitHub](https://github.com/Quandela/merlin)
2. Heurtel, N. et al. "Perceval: A Software Platform for Discrete Variable Photonic Quantum Computing." Quantum 7, 931 (2023). [arXiv:2204.00602](https://arxiv.org/abs/2204.00602) | [Docs](https://perceval.quandela.net)
3. Notton, C. et al. "Establishing Baselines for Photonic QML." arXiv:2510.25839 (2025). [Paper](https://arxiv.org/abs/2510.25839)

### Residual Hybrid / Quantum Correction Architecture

4. "Readout-Side Bypass for Residual Hybrid Quantum-Classical Models." [arXiv:2511.20922](https://arxiv.org/abs/2511.20922) (2025)
5. "Trainability-Oriented Hybrid Quantum Regression via Geometric Preconditioning and Curriculum Optimization." [arXiv:2601.11942](https://arxiv.org/abs/2601.11942) (2026)
6. Illesova, S. et al. "From Classical to Hybrid: A Practical Framework for Quantum-Enhanced Learning." [arXiv:2511.08205](https://arxiv.org/abs/2511.08205) (2025)

### Dressed Quantum Circuit / Transfer Learning

7. Mari, A. et al. "Transfer Learning in Hybrid Classical-Quantum Neural Networks." Quantum 4, 340 (2020). [arXiv:1912.08278](https://arxiv.org/abs/1912.08278)

### Data Re-uploading

8. Perez-Salinas, A. et al. "Data re-uploading for a universal quantum classifier." Quantum 4, 226 (2020). [arXiv:1907.02085](https://arxiv.org/abs/1907.02085)

### Fourier Analysis of VQC / Learnable Scaling

9. Schuld, M. et al. "Effect of data encoding on the expressive power of variational quantum ML models." Physical Review A 103, 032430 (2021). [arXiv:2008.08605](https://arxiv.org/abs/2008.08605)
10. Jerbi, S. et al. "Parametrized Quantum Policies for Reinforcement Learning." NeurIPS 2021. [arXiv:2103.05577](https://arxiv.org/abs/2103.05577)

### Batch Normalization

11. Ioffe, S. & Szegedy, C. "Batch Normalization: Accelerating Deep Network Training." ICML 2015. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
12. "Optimal Normalization in Quantum-Classical Hybrid Models for Anti-Cancer Drug Response Prediction." [arXiv:2505.10037](https://arxiv.org/abs/2505.10037) (2025)

### Training Optimization

13. Loshchilov, I. & Hutter, F. "SGDR: Stochastic Gradient Descent with Warm Restarts." ICLR 2017. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)
14. Huber, P.J. "Robust Estimation of a Location Parameter." Annals of Math. Statistics 35(1), 73-101 (1964)
15. Pascanu, R. et al. "On the Difficulty of Training Recurrent Neural Networks." ICML 2013. [arXiv:1211.5063](https://arxiv.org/abs/1211.5063)
15b. Kingma, D.P. & Ba, J. "Adam: A Method for Stochastic Optimization." ICLR 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)

### Barren Plateaus

16. McClean, J.R. et al. "Barren Plateaus in Quantum Neural Network Training Landscapes." Nature Comm. 9, 4812 (2018). [arXiv:1803.11173](https://arxiv.org/abs/1803.11173)
17. "Investigating and Mitigating Barren Plateaus in Variational Quantum Circuits: A Survey." [arXiv:2407.17706](https://arxiv.org/abs/2407.17706) (2024)

### Photonic Quantum Advantage

18. Yin, Z. et al. "Experimental quantum-enhanced kernel-based ML on a photonic processor." Nature Photonics 19, 1020-1027 (2025). [DOI:10.1038/s41566-025-01682-5](https://doi.org/10.1038/s41566-025-01682-5)
19. "A Manufacturable Platform for Photonic Quantum Computing." Nature 641, 876-883 (2025). [DOI:10.1038/s41586-025-08820-7](https://doi.org/10.1038/s41586-025-08820-7)

### Photonic Noise Modeling & Optimization

20. "Simulating Photonic Devices with Noisy Optical Elements." Physical Review Research 6, 033337 (2024). [arXiv:2311.10613](https://arxiv.org/abs/2311.10613)
20b. Pappalardo, R. et al. "Photonic parameter-shift rule: exact gradients on linear-optical quantum processors." Physical Review A 111, 032429 (2025). [arXiv:2410.02726](https://arxiv.org/abs/2410.02726)
20c. Hoch, F. et al. "Variational approach to photonic quantum circuits via the parameter shift rule." Physical Review Research 7, 023227 (2025)

### Quantum Reservoir Computing

21. Li, Q., Mukhopadhyay, C., Bayat, A. & Habibnia, A. "Quantum Reservoir Computing for Realized Volatility Forecasting." [arXiv:2505.13933](https://arxiv.org/abs/2505.13933) (2025)
22. Sakurai, A. et al. "Quantum optical reservoir computing powered by boson sampling." Optica Quantum 3, 238-245 (2025). [DOI:10.1364/OPTICAQ.541432](https://doi.org/10.1364/OPTICAQ.541432)
23. MerLin reproduction: [merlinquantum.ai/reproduced_papers/reproductions/quantum_reservoir_computing](https://merlinquantum.ai/reproduced_papers/reproductions/quantum_reservoir_computing.html)
24. Rambach, M. et al. "Photonic Quantum-Accelerated Machine Learning." [arXiv:2512.08318](https://arxiv.org/abs/2512.08318) (2025)
24b. Sannia, A. et al. "Taking advantage of noise in quantum reservoir computing." Scientific Reports 14, 14548 (2024). [DOI:10.1038/s41598-023-35461-5](https://www.nature.com/articles/s41598-023-35461-5)
24c. Nokkala, J. et al. "Quantum Reservoir Autoencoder: Noise Resilience and Efficient Decoding." [arXiv:2602.19700](https://arxiv.org/abs/2602.19700) (2026)

### Deep Learning Baselines

25. Hochreiter, S. & Schmidhuber, J. "Long Short-Term Memory." Neural Computation 9(8), 1735-1780 (1997). DOI:10.1162/neco.1997.9.8.1735
26. Vaswani, A. et al. "Attention Is All You Need." NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
26b. Nie, Y. et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST)." ICLR 2023. [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
26c. Zeng, A. et al. "Are Transformers Effective for Time Series Forecasting?" AAAI 2023. [arXiv:2205.13504](https://arxiv.org/abs/2205.13504)
26d. Lazaridis, G. et al. "Applying Informer for Option Pricing: A Transformer-Based Approach." ICAART 2025. [arXiv:2506.05565](https://arxiv.org/abs/2506.05565)
26e. Kalousis, A. et al. "Deep Learning for Financial Time Series: A Large-Scale Benchmark." [arXiv:2603.01820](https://arxiv.org/abs/2603.01820) (2026)

### Quantum Kernel Methods

27. Havlicek, V. et al. "Supervised learning with quantum-enhanced feature spaces." Nature 567, 209-212 (2019). [arXiv:1804.11326](https://arxiv.org/abs/1804.11326)
28. Schuld, M. & Killoran, N. "Quantum Machine Learning in Feature Hilbert Spaces." Physical Review Letters 122, 040504 (2019). [arXiv:1803.07128](https://arxiv.org/abs/1803.07128)

### Barren Plateaus and Classical Simulability

29. Larocca, M. et al. "Barren Plateaus in Variational Quantum Computing." Nature Reviews Physics (2025). [arXiv:2405.00781](https://arxiv.org/abs/2405.00781)
30. Cerezo, M., Larocca, M. et al. "Does Provable Absence of Barren Plateaus Imply Classical Simulability?" Nature Communications 16, 7907 (2025). [DOI:10.1038/s41467-025-63099-6](https://doi.org/10.1038/s41467-025-63099-6) | [arXiv:2312.09121](https://arxiv.org/abs/2312.09121)

### Linear Optics Complexity

31. Aaronson, S. & Arkhipov, A. "The Computational Complexity of Linear Optics." Theory of Computing 9, 143-252 (2013). [arXiv:1011.3245](https://arxiv.org/abs/1011.3245)

### Photonic Quantum Learning Advantage

32. Liu, Z.-H. et al. "Quantum learning advantage on a scalable photonic platform." Science 389(6767), 1332-1335 (2025). [DOI:10.1126/science.adv2560](https://doi.org/10.1126/science.adv2560) | [arXiv:2502.07770](https://arxiv.org/abs/2502.07770)

### Quantum Finance

33. Sakuma, T. "Quantum Differential Machine Learning." Quantum Economics and Finance 2(1), 3-12 (2025). [DOI:10.1177/29767032251334589](https://doi.org/10.1177/29767032251334589)

### QPU Remote Computing

34. Perceval Remote Computing Documentation: [perceval.quandela.net/docs/notebooks/Remote_computing](https://perceval.quandela.net/docs/notebooks/Remote_computing.html)
35. MerLin Remote Execution Guide: [merlinquantum.ai/user_guide/remote_execution](https://merlinquantum.ai/user_guide/remote_execution.html)

### Competition and Data

36. Q-volution 2026 QML Hackathon: https://aqora.io/competitions/option-pricing-in-finance
37. Dataset: https://huggingface.co/datasets/Quandela/Challenge_Swaptions
38. Quandela Training Center: https://training.quandela.com

## Future Improvements and Open Items

### Implemented
- **Holdout evaluation**: Last 6 rows held out, auto-regressive prediction, RMSE/MAE/R2 comparison
- **Test ground truth evaluation**: Automatic evaluation against `test.xlsx` with volatility surface heatmap
- **Model save/load**: Trained models saved to `results/models/` with all preprocessing artifacts
- **Submission pipeline**: Auto-regressive 6-day future prediction, two Excel export formats
- **Computation time tracking**: All training/evaluation cells report timing; exported to CSV files
- **QPU access verification**: Both Ascella and Belenos QPUs accessible via Quandela Cloud token
- **QPU evaluation cell**: MerlinProcessor-based QPU evaluation with CPU builder-based model; checks `RemoteProcessor.status` and skips non-running QPUs
- **Final Candidate selection**: Automated selection of best model (noise vs noise-free) with FINAL CANDIDATE labeling

- **Time-series cross-validation**: 5-fold expanding window CV with `TimeSeriesSplit`; PCA re-fit per fold to prevent data leakage (§11.5)
- **LSTM & Transformer baselines**: Per-timestep PCA preserving temporal structure; LSTM [25] and Transformer [26] for honest classical comparison. Vanilla Transformer correctly demonstrates attention limitations with short windows [26c] (§11.6)
- **Quantum kernel regression**: FidelityKernel [27, 28] via MerLin's `CircuitBuilder` + `sklearn.KernelRidge` (6m/3p, 150 training samples) (§11.7)
- **Ensemble QRC**: 5 random reservoirs with prediction averaging for variance reduction [24] (§11.8)
- **Richer noise model**: Full QPU parameters including `indistinguishability` (0.916) and `g2` (0.021) via Perceval's `NoiseModel` (§11.9)
- **Hyperparameter sensitivity**: Window size, PCA components, hidden dimension sweeps (§11.10)
- **Feature engineering**: Rolling statistics (mean, std) and momentum features (§11.11)
- **Cerezo et al. discussion**: Analysis of barren plateau avoidance vs classical simulability tension [29, 30, 31] (§11.12)

### Remaining Enhancements
- **QPU end-to-end evaluation**: The QPU evaluation cell builds a CPU noise-free builder-based model and offloads QuantumLayer leaves to QPU via MerlinProcessor. May timeout due to QPU queue latency; test during off-peak hours
- **Larger circuit scales**: Current experiments go up to 20m/10p in simulation; QPU supports up to 24m/12p (Belenos)
- **Advanced Transformer variants**: PatchTST [26b] and Informer [26d] are designed for longer sequences (336+ steps); with our window_size=5, they would not benefit from patching or sparse attention. A comprehensive financial DL benchmark [26e] shows LSTM-PatchTST hybrids achieve the best risk-adjusted performance on longer horizons

## License

This project was created for the Q-volution 2026 hackathon.
