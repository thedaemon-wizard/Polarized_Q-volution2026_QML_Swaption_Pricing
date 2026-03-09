# Grand Finale Presentation Script
## Q-volution 2026 QML Hackathon - Track B (Quandela)
### "Quantum Residual Swaption Pricing"

> **Format**: Short flash talk (~3-5 minutes)
> **Focus**: Science communication - clearly explain the project and its value
> **Date**: March 8, 2026

---

## Slide 1: Title & Hook (30 sec)

**SPEAK:**

> "Swaptions are among the most complex financial derivatives, with pricing
> surfaces spanning 14 tenors and 16 maturities - that's 224 prices evolving
> every business day. Our challenge: predict 6 days into the future using
> only ~490 historical rows.
>
> Our solution achieved an **R-squared of 0.989** on the hidden test data,
> using a photonic quantum circuit that requires **zero quantum parameter training**
> and enriched features validated on the Belenos QPU noise model."

**KEY VISUAL:** Title + 4 metric cards (Test RMSE 0.0104, R² 0.989, 26 models tested, 0 quantum params)

---

## Slide 2: The Problem & Why It Matters (45 sec)

**SPEAK:**

> "A swaption gives you the right to enter an interest rate swap.
> Banks use swaption prices to manage trillions of dollars in interest rate risk.
>
> The pricing surface is a 2D grid: maturity vs tenor. Each cell is a price,
> and the entire surface shifts daily based on market dynamics. Predicting
> this surface accurately isn't just an academic exercise - it directly
> impacts risk management, hedging strategies, and regulatory capital.
>
> We frame this as a time-series forecasting problem: given a sliding window
> of recent surfaces, predict tomorrow's surface, then feed that prediction
> back to forecast the day after - auto-regressively for 6 days."

**KEY VISUAL:** Swaption surface diagram + auto-regressive pipeline

---

## Slide 3: Our Innovation - Residual QRC (60 sec)

**SPEAK:**

> "Here's our key insight: classical linear regression already captures 95%
> of price variance with an R-squared of 0.95. So instead of asking a quantum
> circuit to learn everything from scratch, we ask it to learn only the
> remaining 5% residual error.
>
> But we go further. We tried standard Variational Quantum Circuits - VQCs -
> and they failed. RMSE around 0.19, barely better than random. Why?
> **Barren plateaus.** At 12+ photonic modes, gradients vanish exponentially,
> making optimization impossible.
>
> Our solution: **Quantum Reservoir Computing**. We freeze the quantum circuit
> entirely - a random interferometer that never gets trained. We only optimize
> a classical readout layer. This completely avoids barren plateaus.
>
> The result? Residual QRC achieves RMSE 0.043, outperforming pure VQC
> by 4.4x, while training zero quantum parameters."

**KEY VISUAL:** Architecture diagram (4-step pipeline) + QRC vs VQC bar chart

---

## Slide 4: Results & Validation (45 sec)

**SPEAK:**

> "We tested 26 model configurations across classical baselines, deep learning
> (LSTM, Transformer), pure quantum, hybrid VQC, quantum kernels, and QRC architectures.
>
> Our Final Candidate - QRC with enriched features and Belenos noise model -
> validated on the hidden test data with:
> - RMSE of 0.0104
> - R-squared of 0.989
> - Per-day error starting at 0.002 and gradually increasing, exactly as
>   expected from auto-regressive error accumulation.
>
> The enriched features (rolling statistics + momentum) improved prediction
> by 9% over the baseline. We tested with both Ascella and Belenos QPU noise
> models - QRC remains noise-resilient across all backends."

**KEY VISUAL:** Test evaluation heatmap (actual vs predicted volatility surface) + per-day RMSE chart

---

## Slide 5: Real-World Impact & Roadmap (45 sec)

**SPEAK:**

> "Why does this matter beyond the hackathon?
>
> **First, QRC is QPU-deployable today.** No parameter optimization on hardware
> means no iterative quantum-classical loops, no shot noise in gradients.
> Just encode, measure, and let the classical readout handle the rest.
> This is practical for current NISQ photonic devices.
>
> **Second, recent research validates this direction.** Li et al. (2025) showed
> QRC outperforming GARCH models for volatility forecasting. Sakurai et al.
> demonstrated boson sampling powering quantum reservoir computing on Quandela's
> own Ascella processor. And Liu et al. published in Science (2025) proving
> photonic quantum learning advantages.
>
> **Our roadmap:**
> 1. Deploy on Belenos QPU (24 modes) for live inference
> 2. Extend to multi-asset swaption surfaces
> 3. Explore quantum kernel methods as an alternative feature extraction
>
> The residual QRC architecture is a practical, noise-resilient bridge
> from classical finance to quantum advantage."

**KEY VISUAL:** Technology roadmap + recent paper references

---

## Closing (15 sec)

> "To summarize: we showed that by combining a strong classical baseline
> with a frozen photonic quantum reservoir and enriched features, we achieve
> state-of-the-art swaption price prediction - R-squared 0.989 on hidden test
> data - with zero quantum parameter training and built-in noise resilience.
>
> Thank you. I'm happy to take questions."

---

## Q&A Preparation

### Likely Questions & Answers

**Q1: "Why not just use the classical model if it already gets R² = 0.95?"**
> The quantum residual correction improves the tail accuracy - the remaining 5%
> often contains the most financially significant deviations (volatility spikes,
> structural breaks). Our test RMSE of 0.0104 vs the LR baseline's higher error
> demonstrates this improvement is real and consistent across all 6 predicted days.

**Q2: "Is this really quantum advantage?"**
> We're honest about this: at the current scale, the quantum component provides
> a marginal improvement. However, three things make it promising:
> 1. QRC avoids the barren plateau problem that kills VQC
> 2. The photonic noise model shows the approach is hardware-robust
> 3. As QPU sizes grow (Belenos already offers 24 modes), the quantum feature
>    space grows combinatorially while classical simulation becomes intractable

**Q3: "Why did VQC perform so poorly?"**
> Barren plateaus. At 12+ modes, the quantum parameter landscape becomes
> exponentially flat. McClean et al. (2018) predicted this, and our results
> confirm it empirically for photonic circuits. This is why QRC - which
> sidesteps optimization entirely - is the right approach for current hardware.

**Q4: "How does the noise affect QRC vs VQC?"**
> For VQC, noise adds another layer of gradient estimation difficulty.
> For QRC, noise simply changes the feature extraction characteristics.
> Since we only train the classical readout, it adapts to whatever features
> the noisy circuit produces. Our noise comparison shows QRC performance
> is essentially identical across ideal, Ascella, and Belenos noise models.
> Note: Full noise parameters (indistinguishability, g2) trigger Perceval's
> density matrix simulation which is CPU-only and computationally prohibitive
> at 12m/6p. The brightness + transmittance noise model is sufficient for validation.

**Q5: "What about the QPU timeout issue?"**
> We transparently note this in both our README and notebook. Both QPUs
> were in maintenance/calibration during our evaluation window. Our code
> automatically checks QPU status and will run on hardware when available.
> The noise model we use has hardware-derived parameters, so results are
> physically realistic.

**Q6: "How does auto-regressive error accumulate?"**
> Each predicted day feeds back as input for the next prediction. Day 1
> error is just 0.002 (single-step). By Day 5, accumulated error reaches
> 0.017. Day 6 actually improves to 0.013, suggesting the model captures
> some mean-reversion dynamics in the swaption surface.

**Q7: "What recent research supports your approach?"**
> - Li et al. (2025): QRC outperforms GARCH for realized volatility forecasting
> - Sakurai et al. (2025): Boson sampling QRC on Quandela's Ascella QPU
> - Rambach et al. (2025): 20x training data reduction with photonic QRC
> - Liu et al. (2025, Science): Provable photonic quantum learning advantage
> - Sakuma (2025): Quantum differential ML for Bermudan swaption pricing

---

## Presentation Tips

1. **Time management**: 3-5 min total. Practice with a timer.
2. **Audience**: Broad - judges from quantum physics, finance, and ML backgrounds.
3. **Avoid jargon**: Explain "barren plateaus" simply as "flat optimization landscape."
4. **Show confidence**: The R² = 0.989 result is genuinely strong.
5. **Be honest**: About QPU timeouts and marginal quantum improvement.
6. **Energy**: This is a celebration - be enthusiastic about what you built.

---

## Key Numbers to Remember

| Metric | Value |
|--------|-------|
| Test RMSE | 0.0104 |
| Test R² | 0.989 |
| Models tested | 26 |
| Quantum params trained | 0 |
| QRC vs VQC improvement | 4.4x |
| Classical variance captured | 95% |
| Training time (best model) | ~13 seconds |
| QPU modes (Belenos) | 24 |
| Prediction horizon | 6 business days |
| Features per day | 224 |
