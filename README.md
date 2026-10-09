# DrDnaResnet18 (superseded)

> **This repository is the early development copy of the Dr. DNA reproduction and is no longer
> maintained.** Use
> **[Detecting-Silent-Data-Corruptions-in-Deep-Neural-Networks](https://github.com/amanyagami/Detecting-Silent-Data-Corruptions-in-Deep-Neural-Networks)**
> instead: it has a vectorized `drdna` package, a calibrated fault-injection evaluation, tests, CI
> and documented results.

What this repository contains, kept for reference:

| Path | Contents |
|---|---|
| `DRDNA/offlineProfiling.py` | Original tau1/tau2/tau3 profiling (histograms via `plt.hist`) |
| `DRDNA/Onlinedecmiti.py`, `DRDNA/Histogram.py` | Early online detection and mitigation experiments |
| `src/` | CIFAR ResNet-18, pytorchFI-based bit-flip injector |
| `state_dicts/resnet18.pt` | CIFAR-10 ResNet-18 weights (the maintained repo uses the same file via Git LFS) |
| `data/` | CIFAR-10 test data |

Known issues in these scripts (addressed in the maintained repository): histograms are computed
with `plt.hist` (matplotlib in the compute path), and forward hooks are registered without ever
being removed.

Paper: Ma et al., *Dr. DNA: Combating Silent Data Corruptions in Deep Learning using Distribution of
Neuron Activations*, ASPLOS '24, [doi:10.1145/3620666.3651349](https://doi.org/10.1145/3620666.3651349).

Licensed under the MIT License.
