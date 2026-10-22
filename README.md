# Thermal Plate Surrogate

> **Work in progress.** Milestones 1–6 (solver, dataset, POD baseline, neural surrogates, regularization and data-size study) are done; the speed-up benchmark, uncertainty quantification and Docker stages are next.

A neural network that replaces a finite-difference heat-conduction solver: given nine physical parameters, it predicts the **full 2D temperature field** of a plate in a fraction of the solver's time, and reports how uncertain each prediction is.

## The physics

Steady heat conduction on the unit square with uniform conductivity $k$, a Gaussian heat source and fixed edge temperatures:

$$
-k\,\nabla^2 T = q(x,y), \qquad q(x,y) = A\,\exp\!\left(-\frac{(x-x_0)^2+(y-y_0)^2}{2\sigma^2}\right),
$$

with $T = T_\text{left}, T_\text{right}, T_\text{bottom}, T_\text{top}$ on the four edges. The surrogate's inputs are the nine parameters $(k, A, x_0, y_0, \sigma, T_\text{left}, T_\text{right}, T_\text{bottom}, T_\text{top})$.

## The reference solver

`src/thermal_surrogate/solver.py` discretises the equation with the 5-point finite-difference stencil on an $n\times n$ grid ($h = 1/(n-1)$):

$$
\frac{k}{h^2}\left(4T_{i,j} - T_{i+1,j} - T_{i-1,j} - T_{i,j+1} - T_{i,j-1}\right) = q_{i,j},
$$

assembles the sparse system with Kronecker products, $A = k\,(I\otimes D + D\otimes I)$ with $D=\tfrac{1}{h^2}\,\mathrm{tridiag}(-1,2,-1)$, moves the known boundary values to the right-hand side, and solves it with a sparse direct LU factorisation.

**Verification.** Against a manufactured solution $T^* = \sin(2\pi x)\cos(\pi y) + x^2 + y$ (non-zero on every edge, $k = 2.5$), the max-norm error falls as $h^2$:

| $n$ | $h$ | max error | observed order |
|---:|---:|---:|---:|
| 17 | 0.0625 | 5.77e-3 | 1.94 |
| 33 | 0.0313 | 1.44e-3 | 2.00 |
| 65 | 0.0156 | 3.62e-4 | 2.00 |
| 129 | 0.0078 | 9.05e-5 | 2.00 |
| 257 | 0.0039 | 2.26e-5 | 2.00 |

![Solver verification](figures/solver_verification.png)

The test suite also checks that quadratics are reproduced to machine precision, the maximum principle, mirror symmetry, superposition and the exact $1/k$ scaling.

## The dataset

`scripts/make_dataset.py` samples the 9 parameters and solves each case at 64×64 (1100 solves, about 13 s on a laptop):

| Split | Size | How it's sampled |
|---|---:|---|
| train / val / test | 800 / 100 / 100 | one Latin hypercube over the parameter box in `configs/default.yaml`, randomly split |
| out-of-distribution | 100 | Latin hypercube over the box widened by 25% of each range on both sides (clipped to physical limits), keeping only points outside the training box |

A Latin hypercube splits every parameter's range into *N* equal slices and puts exactly one sample in each slice, so each parameter is evenly covered even with few samples in 9 dimensions. Every sample also stores an **OOD distance**: how far, in units of the range width, its furthest coordinate lies outside the training box. Later sections use it to show that surrogate error and uncertainty both grow with distance from the training data.

In 9 dimensions almost the whole widened box lies outside the training box: only $(1/1.5)^9 \approx 2.6\%$ of it is inside. A typical OOD sample therefore leaves the box in about three parameters at once.

Generation is deterministic: the script prints a SHA-256 fingerprint of all arrays, and the same config and seed always give the same fingerprint. The data files are not committed; regenerate them with the script.

![Dataset overview](figures/dataset_overview.png)

## Baseline: POD + ridge regression

Before any neural network, the standard reduced-order model sets the bar. **Proper Orthogonal Decomposition** (POD, i.e. PCA of the fields) finds the orthonormal spatial modes $\phi_r$ that best represent the 800 training fields: they are the left singular vectors of the centred data matrix, and $\lambda_r = s_r^2/N$ is the variance each captures. **Ridge regression** on polynomial features of the parameters then predicts the first $R$ mode coefficients, and the field is rebuilt as $\bar T + \sum_r a_r(\mathbf p)\,\phi_r$.

![POD modes](figures/pod_modes.png)

- **The fields are very compressible.** 7 modes capture 99%, 11 capture 99.9% and 17 capture 99.99% of the variance. Modes 2–4 are edge-temperature patterns: by superposition, the four edge temperatures span exactly four dimensions (checked by a test). Mode 6 is a dipole, $\partial_x$ of the source bump, which is how a linear basis represents a source that moves: $u(x-\delta) \approx u - \delta\,\partial_x u$.
- **Regression, not compression, limits accuracy.** Projecting the test fields onto 17 modes loses only 0.28% (the "oracle"), but predicting the coefficients from the parameters costs about 10× more.
- **Physics-informed features help.** The source term scales exactly as $A/k$ (superposition with zero edges), so adding $A/k$ and $1/k$ as inputs cuts the test error from 4.3% to 2.7%.

Hyperparameters (polynomial degree 1–4, ridge $\alpha$ from $10^{-6}$ to $10^4$) are chosen on the validation set. Degree 4 overfits: 714 features for 800 samples.

| Model | degree | test rel. L2 | test RMSE | peak-T error | hotspot error (plate lengths) | OOD rel. L2 |
|---|---:|---:|---:|---:|---:|---:|
| POD + ridge, generic features | 3 | 4.3% | 2.68° | 2.63° | 0.042 | 17.7% |
| POD + ridge, + A/k, 1/k | 3 | 2.7% | 1.78° | 2.29° | 0.034 | 15.7% |
| projection onto 17 modes (oracle) | — | 0.28% | | | | 0.62% |

![POD + ridge results](figures/pod_ridge_results.png)

**Out of distribution, the regression breaks.** The modes still describe OOD fields well (oracle 0.6%), but the cubic polynomial extrapolates badly. Mean error rises to about 16%, and the worst cases (low conductivity or a source near a corner) overshoot the peak temperature by 2–3×. Error grows with $d_\text{OOD}$. That's the gap the neural surrogate and its uncertainty estimates have to address.

Prediction costs about 0.07 ms per field, against about 12 ms for the solver.

## Neural surrogates (PyTorch)

Two multilayer perceptrons (SiLU activations, AdamW, cosine learning-rate decay, best-on-validation checkpoint), both on standardized parameters plus $A/k$ and $1/k$:

- **POD + MLP**: the network predicts the 17 POD coefficients, replacing ridge. Targets are scaled by one global constant, so the MSE loss equals the squared *field* error (the modes are orthonormal).
- **Direct MLP**: the network predicts all 64×64 = 4096 temperatures.

Width and depth are chosen on the validation set (POD + MLP: width 256, depth 2; direct: width 256, depth 3).

| Model | test rel. L2 | test p95 | test RMSE | peak-T error | OOD rel. L2 |
|---|---:|---:|---:|---:|---:|
| POD + ridge (baseline) | 2.72% | 9.2% | 1.78° | 2.29° | 15.7% |
| POD + MLP | 0.64% | 2.1% | 0.41° | 0.73° | 3.1% |
| POD + MLP, raw inputs only | 0.99% | 2.7% | 0.69° | 1.36° | 3.9% |
| direct MLP | 0.59% | 1.6% | 0.37° | 0.68° | 2.8% |

![MLP results](figures/mlp_results.png)

- **The networks beat the baseline about 4× in distribution and about 5× out of distribution.** The ridge regression was the bottleneck, and an MLP is a far better coefficient predictor. Out of distribution, a SiLU network grows at most linearly in its inputs, while a cubic polynomial grows like $z^3$.
- **Physics features still help, but less.** Without $A/k$ and $1/k$ the POD + MLP error rises from 0.64% to 0.99%; the network learns most of the $1/k$ dependence on its own.
- **Direct vs. POD output.** The direct MLP is slightly more accurate but has about 1M more parameters, trains several times slower, and its errors are pixel-noisy: nothing in its architecture ties neighbouring pixels together. The POD version is smooth by construction.
- **Overfitting.** Training loss ends about 100× below validation loss. With only 800 samples, the networks memorize; the next milestone quantifies how much regularization helps.

## Regularization and data-size study

The POD + MLP was retrained **5 times per configuration with different random seeds** (initialization and minibatch order), and every number below is a mean ± standard deviation over those seeds. Selection uses the mean validation error. The gap is val MSE / train MSE at the kept epoch.

| width | weight decay | val rel. L2 | test rel. L2 | OOD rel. L2 | gap |
|---:|---:|---:|---:|---:|---:|
| 64 | 0 | 1.03 ± 0.07% | 0.94 ± 0.06% | 4.2% | 10× |
| 64 | 0.001 | 1.02 ± 0.07% | 0.93 ± 0.06% | 4.2% | 10× |
| 64 | 0.01 | 1.01 ± 0.07% | 0.92 ± 0.05% | 4.1% | 10× |
| 64 | 0.1 | 1.03 ± 0.05% | 0.94 ± 0.03% | 4.0% | 8× |
| 256 | 0 | 0.75 ± 0.09% | 0.69 ± 0.06% | 3.2% | 56× |
| 256 | 0.001 | 0.78 ± 0.16% | 0.72 ± 0.14% | 3.2% | 66× |
| 256 | 0.01 | 0.70 ± 0.05% | 0.64 ± 0.03% | 3.0% | 44× |
| 256 | 0.1 | 0.73 ± 0.07% | 0.68 ± 0.08% | 2.8% | 13× |

Then, with the best setting (width 256, weight decay 0.01), the training set was cut to random subsets (a different subset per seed):

| training simulations | test rel. L2 | OOD rel. L2 | gap |
|---:|---:|---:|---:|
| 100 | 4.56 ± 0.36% | 12.1% | 6515× |
| 200 | 2.53 ± 0.17% | 7.4% | 754× |
| 400 | 1.25 ± 0.06% | 4.9% | 186× |
| 800 | 0.62 ± 0.02% | 2.9% | 84× |

![Regularization study](figures/regularization.png)

- **Data is the lever, and the scaling is clean.** Test error falls as $n^{-0.97}$: every doubling of the simulation budget halves the error, with no sign of a plateau at 800. With just 200 simulations the MLP already matches POD + ridge trained on 800. This answers the question a customer would ask: how many expensive simulations buy a given accuracy.
- **Weight decay barely matters.** At width 256 its best value (0.01) improves test error from 0.69% to 0.64%, about one seed standard deviation, so it isn't a clear win. What it *does* do is cut the seed-to-seed spread in half, and at 0.1 it shrinks the train/val gap 4× without improving test error.
- **A large gap isn't the problem.** Width 256 overfits 5× more than width 64 (gap 56× vs 10×) yet generalizes better (0.69% vs 0.94%). The gap measures memorization of the training set; what matters is held-out error, and the extra capacity is being used.
- **Seed noise is about ±0.05%.** Differences smaller than that, like the width-256 vs width-512 choice in the previous section, can't be resolved with a single run.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                           # 38 tests (solver, dataset, POD, metrics, MLP, study helpers)
python scripts/verify_solver.py  # regenerates figures/solver_verification.png
python scripts/make_dataset.py   # writes data/plate_dataset.npz (~13 s)
python scripts/plot_dataset.py   # regenerates figures/dataset_overview.png
python scripts/fit_pod.py        # POD + ridge baseline -> results/pod_ridge.json + figures (~10 s)
python scripts/train_mlp.py      # neural surrogates -> results/mlp.json + figure (~10 min on a laptop CPU)
python scripts/regularization_study.py   # 60 trainings -> results/regularization.json + figure (~25-40 min)
```

## Roadmap

- [x] Repo structure, packaging, tests
- [x] Finite-difference solver with $O(h^2)$ convergence test
- [x] Latin-hypercube dataset, including an out-of-distribution split
- [x] POD + ridge regression baseline
- [x] MLP field surrogate
- [x] Regularisation study over several seeds (plus training-set size)
- [ ] Speed-up benchmark
- [ ] Deep-ensemble uncertainty, in- vs. out-of-distribution
- [ ] Dockerised inference CLI

## License

MIT
