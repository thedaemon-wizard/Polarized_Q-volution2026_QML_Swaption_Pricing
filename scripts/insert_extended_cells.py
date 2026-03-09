#!/usr/bin/env python3
"""Insert 18 new cells into swaption_qml.ipynb for extended experiments.

Inserts between existing Cell 33 (QRC) and Cell 34 (§12 header).
"""
import json
import sys

NB_PATH = "swaption_qml.ipynb"

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n") if isinstance(source, str) else source}

def code_cell(source):
    lines = source.split("\n") if isinstance(source, str) else source
    # Add newlines between lines for notebook format
    formatted = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            formatted.append(line + "\n")
        else:
            formatted.append(line)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": formatted
    }

# ============================================================
# CELL DEFINITIONS
# ============================================================

# Cell 0: Configuration
cell_config = code_cell("""\
# ============================================================
# Extended Experiments Configuration
# ============================================================
RUN_EXTENDED = True  # Set False to skip extended experiments

if RUN_EXTENDED:
    from sklearn.model_selection import TimeSeriesSplit
    print("Extended experiments: ENABLED")
    print(f"  Available: TimeSeriesSplit, LSTM, Transformer, Quantum Kernel,")
    print(f"             Ensemble QRC, Rich Noise, Hyperparam Sensitivity,")
    print(f"             Feature Engineering")
else:
    print("Extended experiments: SKIPPED (set RUN_EXTENDED = True to enable)")\
""")

# Cell 1: TS-CV Markdown
cell_tscv_md = md_cell("""\
## 11.5 Time-Series Cross-Validation

### Addressing Data Leakage

The current evaluation uses a **random 80/20 train-test split**, which is problematic for time-series data:
future data points may appear in the training set, inflating validation scores.

We now evaluate using `sklearn.TimeSeriesSplit` with **expanding windows** (5 folds),
where each fold uses only past data for training and immediate future data for validation.
PCA is re-fit on each training fold to prevent information leakage.\
""")

# Cell 2: TS-CV Implementation
cell_tscv_code = code_cell("""\
# ============================================================
# Time-Series Cross-Validation (Expanding Window)
# ============================================================
if RUN_EXTENDED:
    from sklearn.model_selection import TimeSeriesSplit

    N_SPLITS = 5
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    ts_models = {
        "Linear Regression": [],
        "MLP (128,64,32)": [],
        "Residual QRC (Ideal)": [],
    }

    print("=" * 70)
    print(f"Time-Series Cross-Validation ({N_SPLITS} folds)")
    print("=" * 70)

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_seq)):
        t0_fold = time.time()
        X_tr_f, X_va_f = X_seq[train_idx], X_seq[val_idx]
        y_tr_f, y_va_f = y_seq[train_idx], y_seq[val_idx]

        print(f"\\nFold {fold_idx+1}/{N_SPLITS}: train={len(train_idx)}, val={len(val_idx)}")

        # Re-fit PCA per fold (no leakage)
        pca_fold = PCA(n_components=6)
        Xt_f = pca_fold.fit_transform(X_tr_f).astype(np.float32)
        Xv_f = pca_fold.transform(X_va_f).astype(np.float32)

        # --- Linear Regression ---
        lr_fold = LinearRegression()
        lr_fold.fit(Xt_f, y_tr_f)
        lr_pred_f = lr_fold.predict(Xv_f)
        lr_rmse_f = np.sqrt(mean_squared_error(y_va_f, lr_pred_f))
        ts_models["Linear Regression"].append(lr_rmse_f)

        # --- MLP ---
        mlp_fold = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), max_iter=500,
            learning_rate_init=0.005, early_stopping=True,
            validation_fraction=0.15, random_state=SEED)
        mlp_fold.fit(Xt_f, y_tr_f)
        mlp_pred_f = mlp_fold.predict(Xv_f)
        mlp_rmse_f = np.sqrt(mean_squared_error(y_va_f, mlp_pred_f))
        ts_models["MLP (128,64,32)"].append(mlp_rmse_f)

        # --- Residual QRC (Ideal, GPU) ---
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        tr_dl_f = DataLoader(
            TensorDataset(torch.tensor(Xt_f), torch.tensor(y_tr_f.astype(np.float32))),
            batch_size=32, shuffle=True)
        va_dl_f = DataLoader(
            TensorDataset(torch.tensor(Xv_f), torch.tensor(y_va_f.astype(np.float32))),
            batch_size=32, shuffle=False)
        builder_f = build_reservoir_circuit(12, 6)
        ql_f = QuantumLayer(
            input_size=6, builder=builder_f, n_photons=6,
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.UNBUNCHED),
            device=DEVICE, dtype=torch.float32)
        rqrc_fold = ResidualQRCModel(
            lr_fold, ql_f, OUTPUT_SIZE, 6, hidden=64).to(DEVICE)
        tl_f, vl_f = train_model(
            rqrc_fold, tr_dl_f, va_dl_f, DEVICE,
            epochs=30, lr=0.003, use_huber=True, verbose_every=999)
        rqrc_rmse_f = np.sqrt(min(vl_f))
        ts_models["Residual QRC (Ideal)"].append(rqrc_rmse_f)

        elapsed = time.time() - t0_fold
        print(f"  LR={lr_rmse_f:.6f}  MLP={mlp_rmse_f:.6f}  RQRC={rqrc_rmse_f:.6f}  ({elapsed:.1f}s)")

    # Summary
    print("\\n" + "=" * 70)
    print("Time-Series CV Results (mean +/- std)")
    print("=" * 70)
    print(f"{'Model':<30} {'Mean RMSE':>12} {'Std':>10}")
    print("-" * 55)
    for name, scores in ts_models.items():
        mean_s = np.mean(scores)
        std_s = np.std(scores)
        print(f"{name:<30} {mean_s:>12.6f} {std_s:>10.6f}")
    tscv_results = {k: (np.mean(v), np.std(v)) for k, v in ts_models.items()}
else:
    tscv_results = {}
    print("Time-Series CV: SKIPPED")\
""")

# Cell 3: TS-CV Visualization
cell_tscv_viz = code_cell("""\
# Time-Series CV visualization
if RUN_EXTENDED and ts_models:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    ax1 = axes[0]
    box_data = [ts_models[k] for k in ts_models]
    box_labels = list(ts_models.keys())
    bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
    colors = ['#4fc3f7', '#ff8a65', '#81c784']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_ylabel("RMSE")
    ax1.set_title("Time-Series CV Distribution (5 Folds)", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=15)

    # Fold-by-fold comparison
    ax2 = axes[1]
    folds = list(range(1, N_SPLITS + 1))
    for name, scores in ts_models.items():
        ax2.plot(folds, scores, 'o-', label=name, linewidth=2, markersize=6)
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("RMSE")
    ax2.set_title("Per-Fold RMSE Comparison", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(folds)

    plt.tight_layout()
    plt.savefig("results/tscv_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: results/tscv_comparison.png")

    # Compare TS-CV vs random split
    print("\\n--- Random Split vs Time-Series CV ---")
    print(f"  LR  Random: {lr_rmse:.6f}  |  TS-CV: {tscv_results['Linear Regression'][0]:.6f}")
    print(f"  QRC Random: {rqrc_ideal_rmse:.6f}  |  TS-CV: {tscv_results['Residual QRC (Ideal)'][0]:.6f}")\
""")

# Cell 4: LSTM/Transformer Markdown
cell_lstm_md = md_cell("""\
## 11.6 Classical Baselines: LSTM & Transformer

### Deep Time-Series Models

For an honest assessment of quantum advantage, we compare against modern classical
time-series architectures that can exploit temporal structure:

- **LSTM**: Long Short-Term Memory network, the de facto standard for sequence prediction
- **Transformer**: Attention-based model using positional encoding

Both use **per-timestep PCA**: each day's 224 features are reduced to 6 PCA features,
preserving the temporal dimension (window_size=5 timesteps × 6 features).\
""")

# Cell 5: LSTM/Transformer Implementation
cell_lstm_code = code_cell("""\
# ============================================================
# LSTM & Transformer Classical Baselines
# ============================================================
if RUN_EXTENDED:
    # --- Per-timestep PCA ---
    # Fit PCA on individual day vectors (not flattened windows)
    pca_per_step = PCA(n_components=6)
    data_pca = pca_per_step.fit_transform(data_normalized).astype(np.float32)  # (494, 6)

    # Create sequences preserving temporal structure
    def create_sequences_temporal(data_pca, data_raw_norm, window_size=5):
        X, y = [], []
        for i in range(len(data_pca) - window_size):
            X.append(data_pca[i:i+window_size])  # (window, 6)
            y.append(data_raw_norm[i + window_size])  # (224,) next day raw
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    X_temporal, y_temporal = create_sequences_temporal(data_pca, data_normalized, WINDOW_SIZE)
    print(f"Temporal sequences: X={X_temporal.shape}, y={y_temporal.shape}")

    X_tr_t, X_va_t, y_tr_t, y_va_t = train_test_split(
        X_temporal, y_temporal, test_size=0.2, random_state=SEED)

    tr_dl_t = DataLoader(
        TensorDataset(torch.tensor(X_tr_t), torch.tensor(y_tr_t)),
        batch_size=32, shuffle=True)
    va_dl_t = DataLoader(
        TensorDataset(torch.tensor(X_va_t), torch.tensor(y_va_t)),
        batch_size=32, shuffle=False)

    # --- LSTM Model ---
    class LSTMBaseline(nn.Module):
        def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, output_dim=224, dropout=0.1):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=dropout)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim))

        def forward(self, x):
            # x: (batch, window, input_dim)
            lstm_out, (hn, cn) = self.lstm(x)
            return self.head(lstm_out[:, -1, :])  # Last timestep

    # --- Transformer Model ---
    class TransformerBaseline(nn.Module):
        def __init__(self, input_dim=6, d_model=64, nhead=4,
                     num_layers=2, output_dim=224, window_size=5, dropout=0.1):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_encoding = nn.Parameter(
                torch.randn(1, window_size, d_model) * 0.1)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=128,
                dropout=dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, output_dim))

        def forward(self, x):
            x = self.input_proj(x) + self.pos_encoding
            x = self.transformer(x)
            return self.head(x[:, -1, :])

    # --- Train function for temporal models ---
    def train_temporal_model(model, train_dl, val_dl, device, epochs=50, lr=0.003):
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=20, T_mult=2, eta_min=1e-6)
        crit = nn.MSELoss()
        best_vl, best_st = float('inf'), None
        val_losses = []

        for ep in range(epochs):
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                loss = crit(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            model.eval()
            val_mse = []
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    val_mse.append(crit(model(xb), yb).item())
            val_rmse = np.sqrt(np.mean(val_mse))
            val_losses.append(val_rmse ** 2)
            if val_rmse < best_vl:
                best_vl = val_rmse
                best_st = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_st)
        return val_losses, best_vl

    # Train LSTM
    print("=" * 70)
    print("LSTM Baseline Training")
    print("=" * 70)
    torch.manual_seed(SEED)
    t0_lstm = time.time()
    lstm_model = LSTMBaseline(input_dim=6, hidden_dim=64, num_layers=2, output_dim=OUTPUT_SIZE)
    vl_lstm, lstm_rmse = train_temporal_model(
        lstm_model, tr_dl_t, va_dl_t, DEVICE, epochs=50, lr=0.003)
    lstm_time = time.time() - t0_lstm
    print(f"  LSTM RMSE: {lstm_rmse:.6f} ({lstm_time:.1f}s)")
    lstm_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
    print(f"  Parameters: {lstm_params:,}")

    # Train Transformer
    print("\\n" + "=" * 70)
    print("Transformer Baseline Training")
    print("=" * 70)
    torch.manual_seed(SEED)
    t0_tfm = time.time()
    tfm_model = TransformerBaseline(
        input_dim=6, d_model=64, nhead=4, num_layers=2,
        output_dim=OUTPUT_SIZE, window_size=WINDOW_SIZE)
    vl_tfm, tfm_rmse = train_temporal_model(
        tfm_model, tr_dl_t, va_dl_t, DEVICE, epochs=50, lr=0.003)
    tfm_time = time.time() - t0_tfm
    print(f"  Transformer RMSE: {tfm_rmse:.6f} ({tfm_time:.1f}s)")
    tfm_params = sum(p.numel() for p in tfm_model.parameters() if p.requires_grad)
    print(f"  Parameters: {tfm_params:,}")

    print(f"\\n--- Comparison ---")
    print(f"  Linear Regression:   {lr_rmse:.6f}")
    print(f"  MLP (128,64,32):     {mlp_large_rmse:.6f}")
    print(f"  LSTM:                {lstm_rmse:.6f}")
    print(f"  Transformer:         {tfm_rmse:.6f}")
    print(f"  Residual QRC:        {rqrc_ideal_rmse:.6f}")
else:
    lstm_rmse, tfm_rmse = None, None
    lstm_time, tfm_time = 0, 0
    print("LSTM/Transformer: SKIPPED")\
""")

# Cell 6: LSTM/Transformer Visualization
cell_lstm_viz = code_cell("""\
# LSTM/Transformer training curves and comparison
if RUN_EXTENDED and lstm_rmse is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training curves
    ax1 = axes[0]
    epochs_range = range(1, len(vl_lstm) + 1)
    ax1.plot(epochs_range, [np.sqrt(v) for v in vl_lstm], label=f"LSTM ({lstm_rmse:.4f})", linewidth=2)
    ax1.plot(epochs_range, [np.sqrt(v) for v in vl_tfm], label=f"Transformer ({tfm_rmse:.4f})", linewidth=2)
    ax1.axhline(lr_rmse, color="red", linestyle="--", alpha=0.5, label=f"LR ({lr_rmse:.4f})")
    ax1.axhline(rqrc_ideal_rmse, color="green", linestyle="--", alpha=0.5,
                label=f"Residual QRC ({rqrc_ideal_rmse:.4f})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation RMSE")
    ax1.set_title("Deep Classical Baselines - Training Curves", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Bar comparison
    ax2 = axes[1]
    models_bar = ["LR", "MLP", "LSTM", "Transformer", "Res. QRC"]
    rmses_bar = [lr_rmse, mlp_large_rmse, lstm_rmse, tfm_rmse, rqrc_ideal_rmse]
    colors_bar = ['#78909c', '#78909c', '#4fc3f7', '#4fc3f7', '#81c784']
    bars = ax2.bar(models_bar, rmses_bar, color=colors_bar, edgecolor='white', linewidth=0.5)
    ax2.set_ylabel("Validation RMSE")
    ax2.set_title("Classical vs Quantum Comparison", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, rmse in zip(bars, rmses_bar):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{rmse:.4f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("results/deep_classical_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: results/deep_classical_comparison.png")\
""")

# Cell 7: Quantum Kernel Markdown
cell_kernel_md = md_cell("""\
## 11.7 Quantum Kernel Regression (FidelityKernel)

### Photonic Fidelity Kernel

An alternative quantum approach uses the **fidelity kernel**:
$K(x_i, x_j) = |\\langle s | U^\\dagger(x_j) U(x_i) | s \\rangle|^2$

This measures the overlap between quantum states encoding different data points.
We use MerLin's `FidelityKernel` with a `CircuitBuilder`-based feature map,
then apply `sklearn.KernelRidge` for regression with the precomputed quantum kernel matrix.

**Circuit**: 6 modes / 3 photons (for computational feasibility)
**Subsample**: Up to 150 training points (kernel matrix computation is O(n²)).\
""")

# Cell 8: Quantum Kernel Implementation
cell_kernel_code = code_cell("""\
# ============================================================
# Quantum Kernel Regression (FidelityKernel)
# ============================================================
if RUN_EXTENDED:
    from merlin import FeatureMap, FidelityKernel
    from sklearn.kernel_ridge import KernelRidge

    KERNEL_MODES = 6
    KERNEL_PHOT = 3
    KERNEL_FEAT = 6
    MAX_KERNEL_SAMPLES = 150

    print("=" * 70)
    print("Quantum Kernel Regression")
    print("=" * 70)

    # Use PCA(6) features (same as QRC)
    Xt_k = d6[2].transform(X_train).astype(np.float32)  # (n_train, 6)
    Xv_k = d6[2].transform(X_val).astype(np.float32)    # (n_val, 6)

    # Subsample for feasibility
    if len(Xt_k) > MAX_KERNEL_SAMPLES:
        np.random.seed(SEED)
        k_idx = np.random.choice(len(Xt_k), MAX_KERNEL_SAMPLES, replace=False)
        k_idx.sort()
        Xt_k_sub = Xt_k[k_idx]
        yt_k_sub = y_train[k_idx]
        print(f"  Subsampled: {len(Xt_k)} -> {MAX_KERNEL_SAMPLES} training points")
    else:
        Xt_k_sub = Xt_k
        yt_k_sub = y_train

    try:
        # Build feature map using CircuitBuilder
        t0_kernel = time.time()
        kernel_builder = CircuitBuilder(n_modes=KERNEL_MODES)
        kernel_builder.add_angle_encoding(
            modes=list(range(min(KERNEL_FEAT, KERNEL_MODES))), name="px")
        kernel_builder.add_entangling_layer(trainable=False, name="ent1")

        feature_map = FeatureMap(
            builder=kernel_builder,
            input_size=KERNEL_FEAT,
            input_parameters="px",
            trainable_parameters=[],
            device=torch.device("cpu"),
            dtype=torch.float32)

        fidelity_kernel = FidelityKernel(
            feature_map=feature_map,
            n_photons=KERNEL_PHOT,
            computation_space=ComputationSpace.UNBUNCHED)

        # Compute training kernel matrix
        print(f"  Computing training kernel matrix ({len(Xt_k_sub)}x{len(Xt_k_sub)})...")
        Xt_k_tensor = torch.tensor(Xt_k_sub, dtype=torch.float32)
        K_train = fidelity_kernel(Xt_k_tensor).detach().numpy()
        print(f"  Training kernel matrix: {K_train.shape}, time={time.time()-t0_kernel:.1f}s")

        # Compute test kernel matrix
        t0_test = time.time()
        print(f"  Computing test kernel matrix ({len(Xv_k)}x{len(Xt_k_sub)})...")
        Xv_k_tensor = torch.tensor(Xv_k, dtype=torch.float32)
        K_test = fidelity_kernel(Xv_k_tensor, Xt_k_tensor).detach().numpy()
        print(f"  Test kernel matrix: {K_test.shape}, time={time.time()-t0_test:.1f}s")

        # Kernel Ridge Regression
        kr = KernelRidge(kernel='precomputed', alpha=1.0)
        kr.fit(K_train, yt_k_sub)
        kr_pred = kr.predict(K_test)
        kernel_rmse = np.sqrt(mean_squared_error(y_val, kr_pred))
        kernel_time = time.time() - t0_kernel

        print(f"\\n  Quantum Kernel Ridge RMSE: {kernel_rmse:.6f}")
        print(f"  Total computation time: {kernel_time:.1f}s")
        print(f"  Comparison:")
        print(f"    LR:              {lr_rmse:.6f}")
        print(f"    Residual QRC:    {rqrc_ideal_rmse:.6f}")
        print(f"    Quantum Kernel:  {kernel_rmse:.6f}")
    except Exception as e:
        print(f"  Quantum Kernel FAILED: {e}")
        print(f"  This is expected if the kernel computation is too expensive.")
        print(f"  Falling back to classical RBF kernel for comparison...")
        # Fallback: classical RBF kernel
        from sklearn.kernel_ridge import KernelRidge as KRR_fallback
        kr_rbf = KRR_fallback(kernel='rbf', alpha=1.0, gamma=0.1)
        kr_rbf.fit(Xt_k_sub, yt_k_sub)
        kr_rbf_pred = kr_rbf.predict(Xv_k)
        kernel_rmse = np.sqrt(mean_squared_error(y_val, kr_rbf_pred))
        kernel_time = 0
        print(f"  Classical RBF Kernel RMSE: {kernel_rmse:.6f} (fallback)")
else:
    kernel_rmse = None
    kernel_time = 0
    print("Quantum Kernel: SKIPPED")\
""")

# Cell 9: Ensemble QRC Markdown
cell_ensemble_md = md_cell("""\
## 11.8 Ensemble QRC (Multiple Random Reservoirs)

### Variance Reduction via Ensemble

Each QRC model uses a **different random interferometer** as its quantum reservoir.
By averaging predictions from N=5 models, we can reduce prediction variance.
This is analogous to random forest ensembling but with quantum feature extractors.\
""")

# Cell 10: Ensemble QRC Implementation
cell_ensemble_code = code_cell("""\
# ============================================================
# Ensemble QRC (Multiple Random Reservoirs)
# ============================================================
if RUN_EXTENDED:
    N_ENSEMBLE = 5
    ensemble_preds = []
    ensemble_rmses = []

    print("=" * 70)
    print(f"Ensemble QRC ({N_ENSEMBLE} random reservoirs)")
    print("=" * 70)
    t0_ens = time.time()

    for seed_offset in range(N_ENSEMBLE):
        torch.manual_seed(SEED + seed_offset)
        np.random.seed(SEED + seed_offset)

        builder_e = build_reservoir_circuit(QRC_MODES, QRC_FEAT)
        ql_e = QuantumLayer(
            input_size=QRC_FEAT, builder=builder_e, n_photons=QRC_PHOT,
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.UNBUNCHED),
            device=DEVICE, dtype=torch.float32)
        model_e = ResidualQRCModel(
            lr_model, ql_e, OUTPUT_SIZE, QRC_FEAT, hidden=QRC_HIDDEN).to(DEVICE)
        tl_e, vl_e = train_model(
            model_e, d6[0], d6[1], DEVICE,
            epochs=EPOCHS_QRC, lr=0.003, use_huber=True, verbose_every=999)
        rmse_e = np.sqrt(min(vl_e))
        ensemble_rmses.append(rmse_e)

        # Collect predictions
        model_e.eval()
        preds_e = []
        with torch.no_grad():
            for xb, yb in d6[1]:
                preds_e.append(model_e(xb.to(DEVICE)).cpu().numpy())
        ensemble_preds.append(np.vstack(preds_e))
        print(f"  Model {seed_offset+1}/{N_ENSEMBLE}: RMSE={rmse_e:.6f}")

    # Ensemble average
    avg_pred = np.mean(ensemble_preds, axis=0)
    # Get actual validation targets
    va_targets = []
    for xb, yb in d6[1]:
        va_targets.append(yb.numpy())
    va_targets = np.vstack(va_targets)

    ensemble_rmse = np.sqrt(mean_squared_error(va_targets, avg_pred))
    ensemble_time = time.time() - t0_ens

    print(f"\\n  Individual RMSE: {np.mean(ensemble_rmses):.6f} +/- {np.std(ensemble_rmses):.6f}")
    print(f"  Ensemble RMSE:   {ensemble_rmse:.6f}")
    print(f"  Single best QRC: {rqrc_ideal_rmse:.6f}")
    improvement = (rqrc_ideal_rmse - ensemble_rmse) / rqrc_ideal_rmse * 100
    print(f"  Improvement:     {improvement:+.2f}% ({ensemble_time:.1f}s)")
else:
    ensemble_rmse = None
    ensemble_time = 0
    print("Ensemble QRC: SKIPPED")\
""")

# Cell 11: Richer Noise Markdown
cell_noise_md = md_cell("""\
## 11.9 Richer Noise Model (Full QPU Parameters)

### Beyond Brightness + Transmittance

The current noise model uses only **brightness** and **transmittance**.
Real QPU noise also includes **indistinguishability** and **g2** (multi-photon emission probability).

We now compare against a richer `NoiseModel` using all available hardware parameters.\
""")

# Cell 12: Richer Noise Implementation
cell_noise_code = code_cell("""\
# ============================================================
# Richer Noise Model (Full QPU Parameters)
# ============================================================
if RUN_EXTENDED:
    print("=" * 70)
    print("Richer Noise Model Comparison")
    print("=" * 70)

    # Rich noise parameters from QPU data
    rich_noise_configs = {
        'Belenos (simple)': {
            'brightness': noise_params['belenos']['brightness'],
            'transmittance': noise_params['belenos']['transmittance'],
        },
        'Belenos (full)': {
            'brightness': noise_params['belenos']['brightness'],
            'transmittance': noise_params['belenos']['transmittance'],
            'indistinguishability': 0.916,  # From HOM 91.8%
            'g2': 0.021,  # From QPU g2 2.1%
        },
        'Ascella (full)': {
            'brightness': noise_params['ascella']['brightness'],
            'transmittance': noise_params['ascella']['transmittance'],
            'indistinguishability': 0.864,  # From HOM 86.4%
            'g2': 0.0195,  # From QPU g2 1.95%
        },
    }

    rich_noise_results = []

    for config_name, params in rich_noise_configs.items():
        t0_rn = time.time()
        print(f"\\n  {config_name}: {params}")

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # Build noisy circuit
        wl = pcvl.GenericInterferometer(
            QRC_MODES,
            lambda idx: (pcvl.BS() // (0, pcvl.PS(pcvl.P(f"_r{idx}")))),
            shape="rectangle", depth=QRC_MODES)
        for p in wl.get_parameters():
            p.set_value(np.random.uniform(0, 2 * np.pi))

        circ = pcvl.Circuit(QRC_MODES)
        for i in range(min(QRC_FEAT, QRC_MODES)):
            circ.add(i, pcvl.PS(pcvl.P(f"px_{i}")))
        circ = wl // circ // wl

        exp = pcvl.Experiment(circ)
        nm = pcvl.NoiseModel(**params)
        exp.noise = nm
        exp.min_detected_photons_filter(0)
        exp.threshold_detector = True

        ql_rn = QuantumLayer(
            input_size=QRC_FEAT, experiment=exp,
            input_parameters="px", n_photons=QRC_PHOT,
            computation_space=ComputationSpace.UNBUNCHED,
            device=torch.device("cpu"), dtype=torch.float32)

        class NoisyResidualQRC_Rich(nn.Module):
            def __init__(self, lr_model, ql, output_size, n_feat, hidden=64):
                super().__init__()
                self.classical_weight = nn.Parameter(
                    torch.tensor(lr_model.coef_, dtype=torch.float32), requires_grad=False)
                self.classical_bias = nn.Parameter(
                    torch.tensor(lr_model.intercept_, dtype=torch.float32), requires_grad=False)
                self.scale = ScaleLayer(n_feat)
                self.q_layer = ql
                for p in self.q_layer.parameters():
                    p.requires_grad = False
                self.grouping = LexGrouping(ql.output_size, hidden)
                for p in self.grouping.parameters():
                    p.requires_grad = False
                self.bn = nn.BatchNorm1d(hidden)
                self.correction_head = nn.Sequential(
                    nn.ReLU(), nn.Linear(hidden, hidden // 2),
                    nn.ReLU(), nn.Linear(hidden // 2, output_size))
                self.alpha = nn.Parameter(torch.tensor(0.1))
            def forward(self, x):
                base = F.linear(x, self.classical_weight, self.classical_bias)
                xs = self.scale(x)
                with torch.no_grad():
                    qf = self.q_layer(xs)
                    gr = self.grouping(qf)
                corr = self.correction_head(self.bn(gr))
                return base + self.alpha * corr

        model_rn = NoisyResidualQRC_Rich(
            lr_model, ql_rn, OUTPUT_SIZE, QRC_FEAT, hidden=QRC_HIDDEN)

        # CPU DataLoaders
        tr_dl_cpu = DataLoader(
            TensorDataset(d6[0].dataset.tensors[0].cpu(), d6[0].dataset.tensors[1].cpu()),
            batch_size=32, shuffle=True)
        va_dl_cpu = DataLoader(
            TensorDataset(d6[1].dataset.tensors[0].cpu(), d6[1].dataset.tensors[1].cpu()),
            batch_size=32, shuffle=False)

        tl_rn, vl_rn = train_model(
            model_rn, tr_dl_cpu, va_dl_cpu, torch.device("cpu"),
            epochs=30, lr=0.003, use_huber=True, verbose_every=999)
        rmse_rn = np.sqrt(min(vl_rn))
        time_rn = time.time() - t0_rn

        rich_noise_results.append({
            'name': config_name, 'rmse': rmse_rn, 'time': time_rn, 'params': params})
        print(f"  RMSE: {rmse_rn:.6f} ({time_rn:.1f}s)")

    # Summary
    print("\\n" + "=" * 70)
    print("Rich Noise Model Comparison")
    print("=" * 70)
    print(f"{'Config':<25} {'RMSE':>10} {'Time':>8}")
    print("-" * 45)
    print(f"{'Ideal (no noise)':<25} {rqrc_ideal_rmse:>10.6f} {'12s':>8}")
    for r in rich_noise_results:
        print(f"{r['name']:<25} {r['rmse']:>10.6f} {r['time']:>7.1f}s")
    rich_noise_rmse = min(r['rmse'] for r in rich_noise_results)
else:
    rich_noise_results = []
    rich_noise_rmse = None
    print("Rich Noise Model: SKIPPED")\
""")

# Cell 13: Hyperparameter Sensitivity Markdown
cell_hyper_md = md_cell("""\
## 11.10 Hyperparameter Sensitivity Analysis

We sweep key hyperparameters to understand their impact on model performance
and verify robustness of the current configuration.\
""")

# Cell 14: Hyperparameter Sensitivity Implementation
cell_hyper_code = code_cell("""\
# ============================================================
# Hyperparameter Sensitivity Analysis
# ============================================================
if RUN_EXTENDED:
    sensitivity_results = []

    print("=" * 70)
    print("Hyperparameter Sensitivity Analysis")
    print("=" * 70)

    # 1. Window size sweep
    print("\\n--- Window Size Sweep ---")
    for ws in [3, 5, 7, 10]:
        X_ws, y_ws = create_sequences(data_normalized, ws, 1)
        X_tr_ws, X_va_ws, y_tr_ws, y_va_ws = train_test_split(
            X_ws, y_ws, test_size=0.2, random_state=SEED)
        pca_ws = PCA(n_components=6)
        Xt_ws = pca_ws.fit_transform(X_tr_ws).astype(np.float32)
        Xv_ws = pca_ws.transform(X_va_ws).astype(np.float32)

        # LR
        lr_ws = LinearRegression().fit(Xt_ws, y_tr_ws)
        lr_rmse_ws = np.sqrt(mean_squared_error(y_va_ws, lr_ws.predict(Xv_ws)))

        # QRC (Ideal, 30 epochs for speed)
        torch.manual_seed(SEED)
        tr_dl_ws = DataLoader(TensorDataset(
            torch.tensor(Xt_ws), torch.tensor(y_tr_ws.astype(np.float32))),
            batch_size=32, shuffle=True)
        va_dl_ws = DataLoader(TensorDataset(
            torch.tensor(Xv_ws), torch.tensor(y_va_ws.astype(np.float32))),
            batch_size=32, shuffle=False)
        builder_ws = build_reservoir_circuit(12, 6)
        ql_ws = QuantumLayer(
            input_size=6, builder=builder_ws, n_photons=6,
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.UNBUNCHED),
            device=DEVICE, dtype=torch.float32)
        rqrc_ws = ResidualQRCModel(lr_ws, ql_ws, OUTPUT_SIZE, 6, hidden=64).to(DEVICE)
        _, vl_ws = train_model(rqrc_ws, tr_dl_ws, va_dl_ws, DEVICE,
                               epochs=30, lr=0.003, use_huber=True, verbose_every=999)
        rqrc_rmse_ws = np.sqrt(min(vl_ws))

        sensitivity_results.append({
            'param': 'window_size', 'value': ws,
            'lr_rmse': lr_rmse_ws, 'rqrc_rmse': rqrc_rmse_ws})
        print(f"  window={ws}: LR={lr_rmse_ws:.6f}, QRC={rqrc_rmse_ws:.6f}")

    # 2. PCA components sweep
    print("\\n--- PCA Components Sweep ---")
    for n_pca in [4, 6, 8, 10]:
        pca_pc = PCA(n_components=n_pca)
        Xt_pc = pca_pc.fit_transform(X_train).astype(np.float32)
        Xv_pc = pca_pc.transform(X_val).astype(np.float32)

        lr_pc = LinearRegression().fit(Xt_pc, y_train)
        lr_rmse_pc = np.sqrt(mean_squared_error(y_val, lr_pc.predict(Xv_pc)))

        torch.manual_seed(SEED)
        tr_dl_pc = DataLoader(TensorDataset(
            torch.tensor(Xt_pc), torch.tensor(y_train.astype(np.float32))),
            batch_size=32, shuffle=True)
        va_dl_pc = DataLoader(TensorDataset(
            torch.tensor(Xv_pc), torch.tensor(y_val.astype(np.float32))),
            batch_size=32, shuffle=False)
        builder_pc = build_reservoir_circuit(12, min(n_pca, 12))
        ql_pc = QuantumLayer(
            input_size=n_pca, builder=builder_pc, n_photons=6,
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.UNBUNCHED),
            device=DEVICE, dtype=torch.float32)
        rqrc_pc = ResidualQRCModel(lr_pc, ql_pc, OUTPUT_SIZE, n_pca, hidden=64).to(DEVICE)
        _, vl_pc = train_model(rqrc_pc, tr_dl_pc, va_dl_pc, DEVICE,
                               epochs=30, lr=0.003, use_huber=True, verbose_every=999)
        rqrc_rmse_pc = np.sqrt(min(vl_pc))

        sensitivity_results.append({
            'param': 'pca_components', 'value': n_pca,
            'lr_rmse': lr_rmse_pc, 'rqrc_rmse': rqrc_rmse_pc})
        print(f"  PCA={n_pca}: LR={lr_rmse_pc:.6f}, QRC={rqrc_rmse_pc:.6f}")

    # 3. Hidden dimension sweep
    print("\\n--- Hidden Dimension Sweep ---")
    for hidden in [32, 64, 128]:
        torch.manual_seed(SEED)
        builder_h = build_reservoir_circuit(12, 6)
        ql_h = QuantumLayer(
            input_size=6, builder=builder_h, n_photons=6,
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.UNBUNCHED),
            device=DEVICE, dtype=torch.float32)
        rqrc_h = ResidualQRCModel(lr_model, ql_h, OUTPUT_SIZE, 6, hidden=hidden).to(DEVICE)
        _, vl_h = train_model(rqrc_h, d6[0], d6[1], DEVICE,
                              epochs=30, lr=0.003, use_huber=True, verbose_every=999)
        rqrc_rmse_h = np.sqrt(min(vl_h))
        sensitivity_results.append({
            'param': 'hidden_dim', 'value': hidden,
            'lr_rmse': lr_rmse, 'rqrc_rmse': rqrc_rmse_h})
        print(f"  hidden={hidden}: QRC={rqrc_rmse_h:.6f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, param_name, xlabel in zip(axes,
            ['window_size', 'pca_components', 'hidden_dim'],
            ['Window Size', 'PCA Components', 'Hidden Dimension']):
        subset = [r for r in sensitivity_results if r['param'] == param_name]
        vals = [r['value'] for r in subset]
        lr_vals = [r['lr_rmse'] for r in subset]
        qrc_vals = [r['rqrc_rmse'] for r in subset]
        ax.plot(vals, qrc_vals, 'o-', color='#81c784', label='Residual QRC', linewidth=2)
        if param_name != 'hidden_dim':
            ax.plot(vals, lr_vals, 's--', color='#78909c', label='Linear Regression', linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Validation RMSE")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Hyperparameter Sensitivity", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/hyperparameter_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: results/hyperparameter_sensitivity.png")
else:
    sensitivity_results = []
    print("Hyperparameter Sensitivity: SKIPPED")\
""")

# Cell 15: Feature Engineering Markdown
cell_feat_md = md_cell("""\
## 11.11 Feature Engineering

### Rolling Statistics and Momentum

We test whether adding hand-crafted features to the sliding window improves prediction:
- **Rolling mean** and **rolling std** across the window
- **Momentum** (day-over-day change in the last timestep)\
""")

# Cell 16: Feature Engineering Implementation
cell_feat_code = code_cell("""\
# ============================================================
# Feature Engineering (Rolling Statistics + Momentum)
# ============================================================
if RUN_EXTENDED:
    print("=" * 70)
    print("Feature Engineering")
    print("=" * 70)

    def create_enriched_sequences(data, window_size=5):
        X, y = [], []
        for i in range(window_size, len(data) - 1):
            window = data[i-window_size:i]
            flat = window.flatten()  # (window*224,)
            rolling_mean = window.mean(axis=0)  # (224,)
            rolling_std = window.std(axis=0)    # (224,)
            momentum = data[i-1] - data[i-2] if i >= 2 else np.zeros(224)  # (224,)
            enriched = np.concatenate([flat, rolling_mean, rolling_std, momentum])
            X.append(enriched)
            y.append(data[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    X_enriched, y_enriched = create_enriched_sequences(data_normalized, WINDOW_SIZE)
    print(f"  Original features: {X_seq.shape[1]}")
    print(f"  Enriched features: {X_enriched.shape[1]} (+{X_enriched.shape[1] - X_seq.shape[1]})")

    X_tr_e, X_va_e, y_tr_e, y_va_e = train_test_split(
        X_enriched, y_enriched, test_size=0.2, random_state=SEED)
    pca_enriched = PCA(n_components=6)
    Xt_e = pca_enriched.fit_transform(X_tr_e).astype(np.float32)
    Xv_e = pca_enriched.transform(X_va_e).astype(np.float32)

    # LR with enriched features
    lr_enriched = LinearRegression().fit(Xt_e, y_tr_e)
    lr_enriched_rmse = np.sqrt(mean_squared_error(y_va_e, lr_enriched.predict(Xv_e)))

    # QRC with enriched features
    torch.manual_seed(SEED)
    tr_dl_e = DataLoader(TensorDataset(
        torch.tensor(Xt_e), torch.tensor(y_tr_e.astype(np.float32))),
        batch_size=32, shuffle=True)
    va_dl_e = DataLoader(TensorDataset(
        torch.tensor(Xv_e), torch.tensor(y_va_e.astype(np.float32))),
        batch_size=32, shuffle=False)
    builder_e = build_reservoir_circuit(12, 6)
    ql_e2 = QuantumLayer(
        input_size=6, builder=builder_e, n_photons=6,
        measurement_strategy=MeasurementStrategy.probs(
            computation_space=ComputationSpace.UNBUNCHED),
        device=DEVICE, dtype=torch.float32)
    rqrc_e = ResidualQRCModel(lr_enriched, ql_e2, OUTPUT_SIZE, 6, hidden=64).to(DEVICE)
    _, vl_e = train_model(rqrc_e, tr_dl_e, va_dl_e, DEVICE,
                          epochs=30, lr=0.003, use_huber=True, verbose_every=999)
    feat_rqrc_rmse = np.sqrt(min(vl_e))

    print(f"\\n  LR (original):    {lr_rmse:.6f}")
    print(f"  LR (enriched):    {lr_enriched_rmse:.6f}  ({(lr_enriched_rmse - lr_rmse)/lr_rmse*100:+.2f}%)")
    print(f"  QRC (original):   {rqrc_ideal_rmse:.6f}")
    print(f"  QRC (enriched):   {feat_rqrc_rmse:.6f}  ({(feat_rqrc_rmse - rqrc_ideal_rmse)/rqrc_ideal_rmse*100:+.2f}%)")
else:
    lr_enriched_rmse, feat_rqrc_rmse = None, None
    print("Feature Engineering: SKIPPED")\
""")

# Cell 17: Cerezo Discussion Markdown
cell_cerezo_md = md_cell("""\
## 11.12 On Barren Plateaus and Classical Simulability

### Does Avoiding Barren Plateaus Imply Classical Simulability?

A fundamental question raised by Cerezo, Larocca et al. (2025, *Nature Communications* 16, 7907)
challenges the quantum ML community: many commonly used models that **avoid barren plateaus**
can also be **classically simulated efficiently**.

**The tension**: The very structure that enables trainability (encoding into small, classically
simulable subspaces) may undermine the quantum advantage these models aim to provide.

**Relevance to our QRC approach**:

1. Our Residual QRC **explicitly avoids barren plateaus** by freezing all quantum parameters.
   Only the classical readout is trained.

2. The marginal improvement over classical LR (~1%) is consistent with the theoretical
   prediction that trainable-without-BP models may live in classically simulable subspaces.

3. However, this does **not** invalidate the QRC approach for two reasons:
   - **QPU deployment**: QRC requires no iterative quantum-classical optimization loops,
     making it immediately deployable on NISQ photonic hardware
   - **Scaling argument**: At larger mode counts (Belenos: 24 modes), the boson sampling
     feature space grows combinatorially, potentially escaping classical simulability
     (Aaronson & Arkhipov, 2011)

4. The scientific value of this work lies in the **methodology**: demonstrating that
   residual architectures with frozen quantum feature extractors are a practical path
   for near-term quantum devices in financial applications.

**References**:
- Larocca, M. et al. "Barren Plateaus in Variational Quantum Computing." *Nature Reviews Physics* (2025). [arXiv:2405.00781](https://arxiv.org/abs/2405.00781)
- Cerezo, M., Larocca, M. et al. "Does Provable Absence of Barren Plateaus Imply Classical Simulability?" *Nature Communications* 16, 7907 (2025)
- Aaronson, S. & Arkhipov, A. "The Computational Complexity of Linear Optics." *Theory of Computing* 9, 143-252 (2013)\
""")

# ============================================================
# INSERT ALL CELLS
# ============================================================

def main():
    with open(NB_PATH) as f:
        nb = json.load(f)

    # Cells to insert (in order)
    new_cells = [
        cell_config,       # 0: config
        cell_tscv_md,      # 1: TS-CV markdown
        cell_tscv_code,    # 2: TS-CV code
        cell_tscv_viz,     # 3: TS-CV visualization
        cell_lstm_md,      # 4: LSTM markdown
        cell_lstm_code,    # 5: LSTM/Transformer code
        cell_lstm_viz,     # 6: LSTM/Transformer viz
        cell_kernel_md,    # 7: Kernel markdown
        cell_kernel_code,  # 8: Kernel code
        cell_ensemble_md,  # 9: Ensemble markdown
        cell_ensemble_code,# 10: Ensemble code
        cell_noise_md,     # 11: Rich noise markdown
        cell_noise_code,   # 12: Rich noise code
        cell_hyper_md,     # 13: Hyperparam markdown
        cell_hyper_code,   # 14: Hyperparam code
        cell_feat_md,      # 15: Feature eng markdown
        cell_feat_code,    # 16: Feature eng code
        cell_cerezo_md,    # 17: Cerezo discussion
    ]

    # Insert at position 34 (after Cell 33, before existing Cell 34)
    INSERT_POS = 34
    print(f"Inserting {len(new_cells)} cells at position {INSERT_POS}")
    print(f"  Before: {len(nb['cells'])} cells")

    for i, cell in enumerate(new_cells):
        nb['cells'].insert(INSERT_POS + i, cell)

    print(f"  After:  {len(nb['cells'])} cells")

    # Verify syntax of all new code cells
    import ast
    for i, cell in enumerate(new_cells):
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            try:
                ast.parse(src)
                print(f"  Cell {i}: syntax OK")
            except SyntaxError as e:
                print(f"  Cell {i}: SYNTAX ERROR at line {e.lineno}: {e.msg}")
                sys.exit(1)

    # Save
    with open(NB_PATH, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"\nSaved {NB_PATH}")


if __name__ == "__main__":
    main()
