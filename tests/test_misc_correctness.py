# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for vector / types / files fixes."""

import os

import numpy as np
import pytest

from brainmaze_utils.vector import translate, rotate, get_mutual_vectors
from brainmaze_utils.types import ObjDict
from brainmaze_utils.files import get_files


def test_get_mutual_vectors_without_labels():
    # raised UnboundLocalError (used by brainmaze-eeg classifiers without labels)
    x = np.array([[0., 0.], [1., 0.], [0., 2.]])
    v = get_mutual_vectors(x)
    assert v.shape == (9, 2)
    np.testing.assert_array_equal(v[1], x[0] - x[1])
    v2, leg = get_mutual_vectors(x, np.array(['a', 'b', 'c']))
    np.testing.assert_array_equal(v2, v)
    assert leg[1] == 'a-b'


def test_translate_returns_new_float_array():
    x = np.array([[1, 2], [3, 4]])
    y = translate(x, [0.5, -1])
    np.testing.assert_allclose(y, [[1.5, 1], [3.5, 3]])
    np.testing.assert_array_equal(x, [[1, 2], [3, 4]])  # not modified in place


def test_rotate_direction_documented():
    # positive angle rotates clockwise about the points' mean (row-vector convention), as documented
    x = np.array([[0.0, 0.0], [2.0, 0.0]])
    y = rotate(x, 90)
    np.testing.assert_allclose(y, [[1, 1], [1, -1]], atol=1e-12)
    np.testing.assert_array_equal(x, [[0, 0], [2, 0]])


def test_objdict_missing_attribute_does_not_create_key():
    d = ObjDict()
    assert not hasattr(d, 'missing')
    assert 'missing' not in d
    with pytest.raises(AttributeError):
        d.missing
    d.a = 1
    assert d['a'] == 1 and d.a == 1
    d['b']['c'] = 2  # item auto-vivification still works
    assert d.b.c == 2


def test_get_files_only_skips_appledouble(tmp_path):
    (tmp_path / 'sub._run').mkdir()
    for p in ['a.edf', '._a.edf', 'sub._run/b.edf']:
        (tmp_path / p).write_text('x')
    files = get_files(str(tmp_path), ('edf',))
    names = sorted(os.path.relpath(f, tmp_path) for f in files)
    assert names == ['a.edf', os.path.join('sub._run', 'b.edf')]
