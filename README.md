#  Bayesian Regression Models (Linear & Logistic)

This repository contains simple and clean implementations of **Bayesian Linear Regression** and **Bayesian Logistic Regression**, using both **exact inference** and **Laplace Approximation**.

These examples are written in Python using `NumPy`, `SciPy`, and `Matplotlib`, and are designed for learning and teaching purposes.

---

##  Contents

| File | Description |
|------|-------------|
| `bayesian_linear_exact.py` | Standard Bayesian Linear Regression with exact Gaussian posterior |
| `bayesian_linear_laplace.py` | Bayesian Linear Regression using Laplace approximation (optimization-based posterior) |
| `bayesian_logistic_laplace.py` | Bayesian Logistic Regression using Laplace approximation (for classification) |

---

##  Bayesian Linear Regression (Exact)

We model the relationship between input \( \mathbf{x} \in \mathbb{R} \) and output \( y \in \mathbb{R} \) using a polynomial basis:

\[
y = \mathbf{w}^\top \boldsymbol{\phi}(x) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)
\]

### Prior:

\[
p(\mathbf{w}) = \mathcal{N}(\mathbf{0}, \alpha^{-1} \mathbf{I})
\]

### Posterior (closed-form):

\[
\mathbf{S}_N = \left( \alpha \mathbf{I} + \frac{1}{\sigma^2} \Phi^\top \Phi \right)^{-1}, \quad
\mathbf{m}_N = \frac{1}{\sigma^2} \mathbf{S}_N \Phi^\top \mathbf{y}
\]

---

##  Bayesian Linear Regression (Laplace Approximation)

Even though linear regression yields an exact Gaussian posterior, we show how to **recover it via Laplace approximation**:

- Minimize the negative log-posterior to obtain the MAP estimate
- Approximate the posterior with a Gaussian around MAP using the Hessian

This is a useful bridge to generalized models like logistic regression.

---

## Bayesian Logistic Regression (Laplace)

We consider a binary classification task with:

\[
p(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x})), \quad \sigma(z) = \frac{1}{1 + e^{-z}}
\]

Since the posterior is not Gaussian:

\[
p(\mathbf{w} \mid \mathcal{D}) \propto p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) p(\mathbf{w})
\]

we approximate it via Laplace:

- Find the MAP estimate \( \mathbf{w}_{\text{MAP}} \)
- Compute the Hessian at MAP
- Approximate \( p(\mathbf{w} \mid \mathcal{D}) \approx \mathcal{N}(\mathbf{w}_{\text{MAP}}, \mathbf{S}_N) \)

---

##  Sampling and Visualization

Each script samples functions from the posterior distribution and visualizes:

- Posterior mean
- ±2 standard deviation bands
- Posterior predictive samples

Example:

![Posterior samples](figures/bayesian_logistic_posterior.png) *(if added)*

---
ًReference:
Bishop, Pattern Recognition and Machine Learning, 2006
