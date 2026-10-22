# Machine Learning: Math from First Principles

Hands-on notebooks that build the core mathematics of machine learning from scratch in plain Python.

Each notebook takes one concept and works through it in the same order:

1. **Definition**: the formula, its domain and its range.
2. **Implementation**: a from-scratch Python function, written to be numerically stable.
3. **Testing**: sample values, property tests (symmetry, inverses, edge cases) and a check against a reference library.
4. **Visualization**: plots and step-by-step derivations that build intuition.
5. **Where it's used**: how the concept shows up in real ML and deep learning models.

The goal is to understand *why* each formula works and *how* it can break in floating point, not only to call a library function.

---

## Notebooks

| # | Topic | What you'll learn | Open |
|---|-------|-------------------|------|
| 1 | [Sigmoid and logit functions](Sigmoid_and_logit_functions.ipynb) | Stable sigmoid, overflow and `log1p`, symmetry σ(−x) = 1 − σ(x), logit as the inverse of sigmoid, odds and log-odds | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AIwithSakthivel/Machine-Learning/blob/main/Sigmoid_and_logit_functions.ipynb) |

*More topics will be added over time.*

---

## 1. Sigmoid and logit functions

**Sigmoid** maps any real number to a probability:

$$\sigma(x) = \frac{1}{1 + e^{-x}} \qquad \text{Domain: } \mathbb{R}, \quad \text{Range: } (0, 1)$$

**Logit (log-odds)** maps a probability back to a real number. It is the inverse of sigmoid:

$$\text{logit}(p) = \ln\left(\frac{p}{1 - p}\right) \qquad \text{Domain: } (0, 1), \quad \text{Range: } \mathbb{R}$$

**Covered in the notebook:**

- **Numerical stability**
  - Why `1 / (1 + exp(-x))` overflows for large negative `x`.
  - How rewriting it as `exp(x) / (1 + exp(x))` fixes the overflow.
  - Why `math.log1p(-p)` is more precise than `math.log(1 - p)` for tiny `p`.
- **Properties, verified with tests**
  - σ(0) = 0.5
  - σ(−x) = 1 − σ(x)
  - σ′(x) = σ(x)(1 − σ(x))
  - logit(1 − p) = −logit(p)
  - logit(σ(x)) = x
- **Edge cases**
  - Inputs at the extremes, such as ±1000.
  - Invalid probabilities (`p ≤ 0` or `p ≥ 1`) raising a clear `ValueError`.
- **Odds and log-odds**: what p / (1 − p) means, and why the log makes it symmetric around 0.
- **Uses in deep learning**
  - Binary and multi-label classification outputs, with logits paired with `BCEWithLogitsLoss`.
  - LSTM and GRU gates.
  - SiLU/Swish activations and SwiGLU layers.
  - Why sigmoid fell out of use in hidden layers: vanishing gradients.

---

## Getting started

**Run in the browser (recommended):** click the **Open in Colab** badge next to any notebook. Nothing needs to be installed.

**Run locally:**

```bash
git clone https://github.com/AIwithSakthivel/Machine-Learning.git
cd Machine-Learning
pip install numpy scipy matplotlib jupyter
jupyter notebook
```

The core functions use only Python's standard `math` module. NumPy, SciPy and Matplotlib are used for reference checks and plots.

---

## Who this is for

- Data scientists and ML engineers who want a solid grasp of the math behind the models they use.
- Anyone preparing for ML or applied scientist interviews where first-principles reasoning matters.
- Learners who prefer *derive → implement → test → visualize* over memorizing formulas.

---

## Contributing

Found a bug, an unclear explanation or a better visualization? Open an issue or a pull request. Suggestions for new topics are welcome too.

---

## Author

**Sakthivel**: [GitHub](https://github.com/AIwithSakthivel) · [Website](https://aiwithsakthivel.com) · [X / Twitter](https://x.com/AIwithSakthivel)

If you find this useful, consider giving the repo a ⭐.
