# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def kl_divergence(mu1, std1, mu2, std2):
    """
    Parametric KL-Divergence between 2 normal 1-D distributions.

    `Normal Distribution <https://en.wikipedia.org/wiki/Normal_distribution>`_


    """
    return 0.5 * ((std1/std2)**2 + ((mu2-mu1)**2 / std2**2) -1 + 2*np.log(std2/std1))


def kl_divergence_mv(mu1, var1, mu2, var2):
    """
    Multidimensional parametric KL-Divergence between 2 normal distributions.

    `KL-Divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions>`_

    `Trace <https://en.wikipedia.org/wiki/Trace_(linear_algebra)>`_

    """
    return 0.5 * ((np.trace(np.dot(np.linalg.inv(var2), var1))) + np.dot(np.dot((mu2 - mu1), np.linalg.inv(var2)), (mu2-mu1).T) - mu1.shape[1] + np.log(np.linalg.det(var2)/np.linalg.det(var1)))[0, 0]


def combine_gauss_distributions(mu1, std1, N1, mu2, std2, N2):
    """
    Recalculates a normal 1-D distribution given two subsets of data.
    """

    c1 = N1 / (N1 + N2)
    c2 = N2 / (N1 + N2)
    mu_combined = (mu1 * c1) + (mu2 * c2)
    std_combined = np.sqrt(
        (N1*std1**2 + N2*std2**2 + N1*((mu1 - mu_combined)**2) + N2*((mu2 - mu_combined)**2)) / (N1+N2)
    ) #
    # np.sqrt((N1*(std1**2) + N2*(std2**2) + N1*N2*(mu2-mu1)**2/(N1+N2)) / (N1+N2)) # https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2008/086212.pdf - 1.4
    return mu_combined, std_combined


def combine_mvgauss_distributions(mu1, var1, N1, mu2, var2, N2):
    """
    Pooled mean and covariance of two multivariate data subsets from their summaries.

    Exact for population (``ddof=0``) statistics: if ``mu_i``/``var_i`` are the mean
    and the biased covariance (``numpy.cov(..., bias=True)``) of subset ``i``, the
    result equals the mean and biased covariance of the concatenated data:

    ``Sigma = (N1*Sigma1 + N2*Sigma2) / N + N1*N2 / N**2 * (mu2 - mu1)(mu2 - mu1)^T``

    with ``N = N1 + N2``.

    Parameters
    ----------
    mu1, mu2 : array_like
        Means, shape ``(d,)`` or ``(1, d)``.
    var1, var2 : array_like
        Covariance matrices, shape ``(d, d)``.
    N1, N2 : int or float
        Number of samples in each subset.

    Returns
    -------
    mu_combined : numpy.ndarray
        Pooled mean, same shape as ``mu1``.
    var_combined : numpy.ndarray
        Pooled ``(d, d)`` covariance (symmetric).

    Notes
    -----
    .. note:: **Changed after v2.0.0:**
       The between-group term used the element-wise square ``(mu2 - mu1)**2``
       instead of the outer product, so off-diagonal (cross-covariance) terms were
       wrong - e.g. +1.5 instead of the true -1.0 for groups whose means move in
       opposite directions.
    """
    mu1 = np.asarray(mu1, dtype=float)
    mu2 = np.asarray(mu2, dtype=float)
    var1 = np.asarray(var1, dtype=float)
    var2 = np.asarray(var2, dtype=float)
    N = N1 + N2
    c1 = N1 / N
    c2 = N2 / N
    mu_combined = (mu1 * c1) + (mu2 * c2)
    d = (mu2 - mu1).reshape(-1)
    var_combined = c1 * var1 + c2 * var2 + (N1 * N2 / N ** 2) * np.outer(d, d)
    return mu_combined, var_combined


def kl_divergence_nonparametric(pk, qk, eps=None):
    """
    KL divergence ``D(P || Q)`` between discrete distributions (e.g. histograms with
    identical bins), in nats.

    The **last axis** holds the bins. A 1-D input is one distribution; an N-D input
    is a stack of distributions (e.g. ``(n_features, n_bins)``, one histogram per
    row), and the result is the **sum of the per-row divergences** (as in v2.0.0).
    Each distribution is normalised to sum to 1 along the last axis first, so raw
    counts may be passed. Bins with ``p == 0`` contribute 0.

    Parameters
    ----------
    pk, qk : array_like
        Non-negative, finite weights/counts with the same shape, bins on the last
        axis. Every distribution (row) must have positive total mass.
    eps : float, optional
        If ``None`` (default) no smoothing is applied and the result is ``inf``
        whenever some bin has ``p > 0`` and ``q == 0`` (P is not absolutely
        continuous w.r.t. Q). If given (finite, ``> 0``), ``eps`` is added to **every**
        bin of both distributions (after normalisation) and they are re-normalised,
        giving a finite, smoothed estimate.

    Returns
    -------
    float
        ``sum(p * log(p / q))`` over all bins (and rows), ``>= 0``; ``inf`` as
        described above.

    Raises
    ------
    ValueError
        If the shapes differ, any value is negative or not finite, a distribution
        has zero total mass, or ``eps`` is not a finite positive number.

    Notes
    -----
    .. note:: **Changed after v2.0.0:**
       Bins with ``q == 0, p > 0`` were silently dropped (returning e.g. 0.0 instead
       of inf). Inputs were not normalised: for already normalised histograms the
       result is unchanged (also for 2-D stacks of normalised rows, which is how
       brainmaze-eeg calls it), but **raw counts now give the KL divergence of the
       normalised histograms** instead of ``sum(c_p * log(c_p / c_q))``, which
       scaled with the number of samples and was not a divergence. Invalid input
       (negative, NaN/inf, zero mass, different shapes) now raises ``ValueError``.
    """
    p = np.atleast_1d(np.asarray(pk, dtype=float))
    q = np.atleast_1d(np.asarray(qk, dtype=float))
    if p.shape != q.shape:
        raise ValueError(f'pk and qk must have the same shape, got {p.shape} and {q.shape}')
    if np.any(p < 0) or np.any(q < 0) or not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        raise ValueError('pk and qk must be finite and non-negative')
    p_mass = p.sum(axis=-1, keepdims=True)
    q_mass = q.sum(axis=-1, keepdims=True)
    if np.any(p_mass <= 0) or np.any(q_mass <= 0):
        raise ValueError('every distribution in pk and qk (each row along the last axis) must have positive total mass')
    p = p / p_mass
    q = q / q_mass
    if eps is not None:
        if not (np.isfinite(eps) and eps > 0):
            raise ValueError(f'eps must be a finite number > 0, got {eps!r}')
        n_bins = p.shape[-1]
        p = (p + eps) / (1 + eps * n_bins)
        q = (q + eps) / (1 + eps * n_bins)
    support = p > 0
    if np.any(q[support] == 0):
        return np.inf
    return float(np.sum(p[support] * np.log(p[support] / q[support])))
