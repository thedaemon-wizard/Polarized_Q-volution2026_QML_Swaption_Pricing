# Option Pricing with Quantum Machine Learning

## Q-volution 2026 QML Hackathon - Track B (Quandela)

This repository contains our submission for the Q-volution 2026 Quantum Machine Learning Hackathon,
Track B organized by Quandela in collaboration with Mila and AMF (Autorite des marches financiers du Quebec).

The goal is to build a QML model to predict actual prices of put and call options (swaptions)
using the provided training dataset, built with Quandela's MerLin framework for photonic
quantum machine learning.

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

**No test data is provided.** Participants must create their own train/test split from
the ~490 rows. We validate by holding out the last 6 rows as a proxy test set.

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

This approach outperforms the classical baseline (RMSE 0.0432 vs 0.0437).

### Quantum Reservoir Computing (QRC)

We also explore **Quantum Reservoir Computing** where the quantum circuit parameters are
**fixed** (random, never trained) and only a classical readout is optimized. This avoids
barren plateaus entirely and achieves competitive results:

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

The Residual QRC (Belenos noise) achieves the **best overall RMSE (0.0432)**, matching or
outperforming both the trainable VQC and classical baselines.

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
| **Training** | CosineAnnealing LR, Huber loss, gradient clipping |
| **Max Scale** | 20 modes / 10 photons (184,756 quantum output dim) |
| **QPU Noise** | Hardware-derived from Ascella and Belenos QPUs |

### Preprocessing Pipeline

1. **Min-Max normalization** to [0, 1] range (optimal for angle encoding)
2. **Sliding window** (size 5) for time-series prediction
3. **PCA** dimensionality reduction (4-10 components, 98-99.8% variance retained)
4. **Learnable ScaleLayer** instead of fixed pi multiplication

### Circuit Design

- **Standard Circuit**: Entangling -> Angle Encoding -> [Rotations + Entangling] x 2 -> Superpositions
- **Data Re-uploading Circuit**: Superpositions -> [Encoding + Rotations + Entangling] x N stages
- **QRC Reservoir Circuit**: Fixed Random Interferometer -> Encoding -> Fixed Random Interferometer (no trainable quantum params)
- UNBUNCHED computation space (at most 1 photon per mode)
- "Dressed quantum circuit" pattern with classical pre/post-processing

## Results

### Key Finding: Residual Hybrid Model Outperforms Classical Baseline

| Model | RMSE | R2 | Type |
|-------|------|-----|------|
| Linear Regression | 0.0437 | 0.950 | Classical |
| MLP (64, 32) | 0.0507 | -- | Classical |
| MLP (128, 64, 32) | 0.0489 | -- | Classical |
| Standard Quantum (best) | 0.1883 | -- | Quantum only |
| Data Re-uploading 2x | 0.1929 | -- | Quantum only |
| Residual Hybrid (VQC) | 0.0432 | 0.950 | Quantum + Classical |
| QRC Pure (Ideal) | 0.1871 | -- | Reservoir only |
| Residual QRC (Ideal) | 0.0433 | -- | Reservoir + Classical |
| Residual QRC (Ascella) | 0.0432 | -- | Reservoir + Classical + Noise |
| **Residual QRC (Belenos)** | **0.0432** | -- | **Reservoir + Classical + Noise** |

The Residual Hybrid and Residual QRC models achieve the best overall RMSE, demonstrating that quantum
circuits can provide meaningful corrections to classical predictions. Notably, QRC achieves
comparable performance without training any quantum parameters, avoiding barren plateaus.

### Computation Time Summary

All training times measured on NVIDIA RTX PRO 6000 (GPU) with CUDA 12.x.
Noise models run on CPU (MerLin's `PhotonLossTransform` device constraint).

| Backend | Time | Notes |
|---------|------|-------|
| Classical (LR + MLP) | < 1s | CPU only |
| Quantum simulator (GPU) | 14-177s | Scales with mode count |
| Noise model (CPU) | 45-97s | QPU-derived parameters |
| QRC Ideal (GPU) | 13s | No quantum params trained |
| QRC Noisy (CPU) | 148-152s | Fixed circuit + noise |
| Best model retrain | ~13s | 50-80 epochs |

Timing is included in all print outputs and exported to `model_comparison.csv` (Time_s column)
and `noise_comparison.csv` (Time_s column).

### Quantum Circuit Scale Comparison

| Config | Modes | Photons | Q-Output Dim | RMSE | Time |
|--------|-------|---------|-------------|------|------|
| Small | 6 | 3 | 20 | 0.1925 | 14s |
| Medium | 10 | 5 | 252 | 0.1901 | 36s |
| QPU-Ascella | 12 | 6 | 924 | 0.2047 | 54s |
| **Large** | **16** | **8** | **12,870** | **0.1878** | **89s** |
| Max | 20 | 10 | 184,756 | 0.1890 | 177s |

### Data Re-uploading Experiment

| Stages | RMSE | Notes |
|--------|------|-------|
| 1 (standard) | 0.1962 | Baseline |
| 2 | 0.1929 | -1.7% improvement |
| 3 | 0.2133 | Diminishing returns |

### Noise Model Comparison (QPU-Derived Parameters)

| QPU/Config | Brightness | Transmittance | RMSE | Time |
|------------|-----------|--------------|------|------|
| Ascella | 0.1033 | 0.2440 | 0.1833 | 87s |
| Belenos | 0.2337 | 0.5200 | 0.1842 | 97s |
| Ideal | 1.0000 | 1.0000 | 0.1920 | 45s |

Noise parameters derived from live QPU hardware metrics (HOM, Transmittance, g2).

### Quantum Reservoir Computing (QRC) Results

| Model | RMSE | Time | Noise | Quantum Params Trained |
|-------|------|------|-------|----------------------|
| QRC Pure (Ideal) | 0.1871 | 13s | None | 0 (fixed) |
| Residual QRC (Ideal) | 0.0433 | 13s | None | 0 (fixed) |
| Residual QRC (Ascella) | 0.0432 | 152s | Ascella QPU | 0 (fixed) |
| **Residual QRC (Belenos)** | **0.0432** | **148s** | Belenos QPU | **0 (fixed)** |

QRC uses fixed random interferometers as quantum feature extractors, training only
the classical readout layer. Despite no quantum parameter optimization, the Residual QRC
matches and slightly outperforms the fully trainable Residual VQC.

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
- Performance: HOM 91.8%, Transmittance 5.20%, g2 2.1%

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
  Experiment-based (noisy) QuantumLayers cannot be offloaded to QPU.

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
|   |-- results_summary.png            #   Summary visualization (3-panel)
|   |-- holdout_evaluation.png         #   Holdout 6-day auto-regressive evaluation
|   |-- qpu_evaluation.png            #   QPU hardware evaluation (when token set)
|   |-- qpu_evaluation.csv            #   QPU evaluation metrics (when token set)
|   |-- missing_data_analysis.png      #   Level 2 missing data pattern analysis
|   |-- predictions_val.csv            #   Validation set predictions
|   |-- model_comparison.csv           #   All model RMSE comparison
|   |-- noise_comparison.csv           #   Noise model comparison data
|   |-- submission_predictions.xlsx    #   Submission file (test_template format)
|   |-- submission_simulated.xlsx      #   Submission file (sample_Simulated format)
|   |-- models/                        #   Saved trained models
|   |   |-- model_best.pt             #     Best model state dict + metadata
|   |   |-- scaler.pkl                #     Fitted MinMaxScaler
|   |   |-- pca_6.pkl                 #     Fitted PCA(6) transform
|-- sample_datasets/                   # Sample data files for submission format
|   |-- test_template.xlsx             #   Test template (Future prediction + Missing data)
|   |-- train.xlsx                     #   Training data (Excel format)
|   |-- sample_Simulated_Swaption_Price.xlsx  # Example submission format
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

### Model Save/Load

Trained models are saved to `results/models/` for reuse without retraining:
- `model_best.pt`: Model state dict + metadata (class, config, column names)
- `scaler.pkl`: Fitted MinMaxScaler for de-normalization
- `pca_6.pkl`: Fitted PCA(6) transform for dimensionality reduction

### Holdout Evaluation

Since no test data is provided, we validate the auto-regressive prediction by:
1. Holding out the **last 6 rows** of the ~490-row training set
2. Auto-regressively predicting them using only the preceding rows
3. Computing RMSE/MAE/R2 against actual values as a proxy for assessment performance

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

### Barren Plateaus

16. McClean, J.R. et al. "Barren Plateaus in Quantum Neural Network Training Landscapes." Nature Comm. 9, 4812 (2018). [arXiv:1803.11173](https://arxiv.org/abs/1803.11173)
17. "Investigating and Mitigating Barren Plateaus in Variational Quantum Circuits: A Survey." [arXiv:2407.17706](https://arxiv.org/abs/2407.17706) (2024)

### Photonic Quantum Advantage

18. Yin, Z. et al. "Experimental quantum-enhanced kernel-based ML on a photonic processor." Nature Photonics 19, 1020-1027 (2025). [DOI:10.1038/s41566-025-01682-5](https://doi.org/10.1038/s41566-025-01682-5)
19. "A Manufacturable Platform for Photonic Quantum Computing." Nature 641, 876-883 (2025). [DOI:10.1038/s41586-025-08820-7](https://doi.org/10.1038/s41586-025-08820-7)

### Photonic Noise Modeling

20. "Simulating Photonic Devices with Noisy Optical Elements." Physical Review Research 6, 033337 (2024). [arXiv:2311.10613](https://arxiv.org/abs/2311.10613)

### Quantum Reservoir Computing

21. "Quantum Reservoir Computing for Realized Volatility Forecasting." (Resource folder PDF)
22. Sakurai, A. et al. "Quantum optical reservoir computing powered by boson sampling." Optica Quantum 3, 238-245 (2025). [DOI:10.1364/OPTICAQ.541432](https://doi.org/10.1364/OPTICAQ.541432)
23. MerLin reproduction: [merlinquantum.ai/reproduced_papers/reproductions/quantum_reservoir_computing](https://merlinquantum.ai/reproduced_papers/reproductions/quantum_reservoir_computing.html)

### Competition and Data

24. Q-volution 2026 QML Hackathon: https://aqora.io/competitions/option-pricing-in-finance
25. Dataset: https://huggingface.co/datasets/Quandela/Challenge_Swaptions
26. Quandela Training Center: https://training.quandela.com

## Future Improvements and Open Items

### Implemented
- **Holdout evaluation**: Last 6 rows held out, auto-regressive prediction, RMSE/MAE/R2 comparison
- **Model save/load**: Trained models saved to `results/models/` with all preprocessing artifacts
- **Submission pipeline**: Auto-regressive 6-day future prediction, two Excel export formats
- **Computation time tracking**: All training/evaluation cells report timing; exported to CSV files
- **Best model selection fix**: When a noisy QRC model (e.g., Belenos) outperforms noise-free, the final RMSE and training losses correctly reflect the selected model's values
- **QPU access verification**: Both Ascella and Belenos QPUs accessible via Quandela Cloud token
- **QPU evaluation cell**: MerlinProcessor-based QPU evaluation with CPU builder-based model (auto-enabled via `QUANDELA_TOKEN`); checks `RemoteProcessor.status` and skips non-running QPUs

### Potential Model Enhancements
- **QPU end-to-end evaluation**: The QPU evaluation cell builds a CPU noise-free builder-based model and offloads QuantumLayer leaves to QPU via MerlinProcessor. May timeout due to QPU queue latency; test during off-peak hours
- **Quantum kernel methods**: MerLin supports Gaussian kernels and random kitchen sinks (not yet explored)
- **RNN/LSTM baseline**: The training materials suggest RNN as a classical baseline for time-series data
- **Larger circuit scales**: Current experiments go up to 20m/10p in simulation; QPU supports up to 24m/12p (Belenos)
- **Richer noise model**: Current noise uses brightness + transmittance; could add `indistinguishability`, `g2`, dark counts from Perceval's `NoiseModel`
- **Time-series cross-validation**: Current split is random; sequential splits would be more rigorous for time-series forecasting

## License

This project was created for the Q-volution 2026 hackathon.
