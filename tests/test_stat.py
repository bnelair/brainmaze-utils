# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

from brainmaze_utils.stat import (
    combine_gauss_distributions, combine_mvgauss_distributions, kl_divergence_nonparametric,
)


def test_combine_gauss_matches_pooled():
    rng = np.random.default_rng(0)
    a = rng.normal(1, 2, 1000)
    b = rng.normal(-3, 0.5, 300)
    mu, std = combine_gauss_distributions(a.mean(), a.std(), a.size, b.mean(), b.std(), b.size)
    ab = np.concatenate((a, b))
    assert mu == pytest.approx(ab.mean())
    assert std == pytest.approx(ab.std())


def test_combine_mvgauss_exact_pooled_covariance():
    # element-wise square instead of outer product gave +1.0 instead of -1.0 off-diagonal
    rng = np.random.default_rng(0)
    a = rng.multivariate_normal([0, 0, 1], [[1, .3, 0], [.3, 2, .1], [0, .1, .5]], 500)
    b = rng.multivariate_normal([2, -2, 0], np.eye(3), 800)
    mu, cov = combine_mvgauss_distributions(
        a.mean(0), np.cov(a.T, bias=True), len(a), b.mean(0), np.cov(b.T, bias=True), len(b))
    ab = np.vstack((a, b))
    np.testing.assert_allclose(mu, ab.mean(0), atol=1e-12)
    np.testing.assert_allclose(cov, np.cov(ab.T, bias=True), atol=1e-12)
    assert cov[0, 1] < 0  # means move in opposite directions -> negative cross-covariance


def test_kl_divergence_inf_and_normalisation():
    p = np.array([.5, .5, 0])
    q = np.array([.5, 0, .5])
    assert kl_divergence_nonparametric(p, q) == np.inf  # was 0.0
    # counts are normalised: [1,1] vs [1,3] -> 0.5 ln(0.5/0.25) + 0.5 ln(0.5/0.75)
    expected = 0.5 * np.log(2) + 0.5 * np.log(2 / 3)
    assert kl_divergence_nonparametric([1., 1.], [1., 3.]) == pytest.approx(expected)
    assert kl_divergence_nonparametric([2., 2.], [3., 9.]) == pytest.approx(expected)
    assert kl_divergence_nonparametric(p, p) == 0.0


def test_kl_divergence_eps_smoothing():
    p = np.array([.5, .5, 0])
    q = np.array([.5, 0, .5])
    d = kl_divergence_nonparametric(p, q, eps=1e-3)
    assert np.isfinite(d) and d > 1
    with pytest.raises(ValueError):
        kl_divergence_nonparametric([1, -1], [1, 1])
    with pytest.raises(ValueError):
        kl_divergence_nonparametric([1, 1], [1, 1, 1])


def test_kl_2d_is_sum_of_row_kls_v2_scale():
    # brainmaze-eeg passes (n_features, n_bins) stacks of normalised histograms (review R5):
    # the value must equal v2.0.0's sum(p * log(p / q)) over all rows, not that / n_features
    rng = np.random.default_rng(0)
    P = rng.uniform(0.1, 1, (4, 200))
    Q = rng.uniform(0.1, 1, (4, 200))
    P /= P.sum(axis=1, keepdims=True)
    Q /= Q.sum(axis=1, keepdims=True)
    v2 = np.nansum(P * np.log(P / Q))
    assert kl_divergence_nonparametric(P, Q) == pytest.approx(v2, rel=1e-12)
    assert kl_divergence_nonparametric(P, Q) == pytest.approx(
        sum(kl_divergence_nonparametric(P[i], Q[i]) for i in range(4)), rel=1e-12)
    # raw counts per row are normalised per row
    assert kl_divergence_nonparametric(P * 1000, Q * 7) == pytest.approx(v2, rel=1e-12)
    with pytest.raises(ValueError):
        kl_divergence_nonparametric(P, Q[:, :100])
    Z = P.copy()
    Z[2] = 0
    with pytest.raises(ValueError):  # one empty row
        kl_divergence_nonparametric(Z, Q)


@pytest.mark.parametrize('eps', [np.nan, np.inf, -1e-3, 0.0])
def test_kl_eps_must_be_finite_positive(eps):
    # Copilot: eps=nan/inf used to slip through and return a non-finite/0 result
    with pytest.raises(ValueError):
        kl_divergence_nonparametric([1, 0], [0, 1], eps=eps)
