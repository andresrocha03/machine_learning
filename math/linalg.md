# Linear Algebra

## 3Blue1Brown
One of the best youtube channels about math and computer science. This playlist is really worth for being introduced or remembering linear algebra concepts. [Link](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab).

## SVD

### Singular Value Decomposition — Synthesis

#### Definition

For any matrix  

$$
X \in \mathbb{R}^{m \times n},
$$

the Singular Value Decomposition (SVD) is

$$
X = U \Sigma V^T
$$

where:

- $U \in \mathbb{R}^{m \times m}$ has orthonormal columns  
  $$
  U^T U = I
  $$
- $V \in \mathbb{R}^{n \times n}$ has orthonormal columns  
  $$
  V^T V = I
  $$
- $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal with nonnegative entries  
  $$
  \Sigma = \mathrm{diag}(\sigma_1, \dots, \sigma_r)
  $$
  with
  $$
  \sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r \ge 0
  $$

The numbers $\sigma_i$ are called **singular values**.


#### Relation to Spectral Decomposition

SVD is closely related to eigen-decomposition.

Consider:

$$
X^T X
$$

This matrix is:

- Symmetric  
- Positive semidefinite  

By the spectral theorem:

$$
X^T X = V \Lambda V^T
$$

where:

- $V$ contains orthonormal eigenvectors  
- $\Lambda$ contains eigenvalues  

Using SVD:

$$
X^T X = V \Sigma^2 V^T
$$

Therefore:

$$
\lambda_i = \sigma_i^2
$$

So:

- Right singular vectors $v_i$ are eigenvectors of $X^T X$
- Singular values squared are eigenvalues of $X^T X$

Similarly:

$$
X X^T = U \Sigma^2 U^T
$$

So:

- Left singular vectors $u_i$ are eigenvectors of $X X^T$

Thus SVD contains spectral decomposition inside it.

#### Stretching Property (Geometric Interpretation)

SVD decomposes a linear transformation into rotation, stretching and rotation. Indeed:

$$
X = U \Sigma V^T
$$

Applied to a vector $x$:

1. $V^T$ rotates the vector
2. $\Sigma$ stretches it along coordinate axes
3. $U$ rotates it again

#### Connection to PCA and Covariance

If $X$ is a centered data matrix (rows = data points), the covariance matrix is:

$$
C = \frac{1}{n} X^T X
$$

PCA solves:

$$
\max_{\|v\|=1} \mathrm{Var}(Xv)
$$

But:

$$
\mathrm{Var}(Xv) = \frac{1}{n} \|Xv\|^2
= \frac{1}{n} v^T X^T X v
$$

Thus PCA directions are eigenvectors of $X^T X$.

Since:

$$
X^T X = V \Sigma^2 V^T
$$

PCA directions are exactly the right singular vectors of $X$.

Variance explained by each principal component is proportional to:

$$
\sigma_i^2
$$
