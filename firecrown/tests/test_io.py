import os
import numpy as np
import pandas as pd

from ..io import write_statistics


def test_write_statistics(tmpdir):
    analysis_id = 'abc'
    output_path = tmpdir
    odir = os.path.join(output_path, 'output_abc')
    arr1 = pd.DataFrame(
        {'g': [0, 1, 2], 'f': [3.0, -1.0, -2.0]}).to_records(index=False)
    arr2 = pd.DataFrame(
        {'i': [5, 1, 2], 'h': [13.0, -1.0, -2.0]}).to_records(index=False)
    stats = {'a': {'b': arr1}, 'c': arr2}

    write_statistics(analysis_id, output_path, stats)

    assert os.path.exists(os.path.join(odir, 'a', 'b.csv'))
    df = pd.read_csv(os.path.join(odir, 'a', 'b.csv'))
    assert np.array_equal(df['g'], arr1['g'])
    assert np.array_equal(df['f'], arr1['f'])
    assert os.path.exists(os.path.join(odir, 'c.csv'))
    df = pd.read_csv(os.path.join(odir, 'c.csv'))
    assert np.array_equal(df['i'], arr2['i'])
    assert np.array_equal(df['h'], arr2['h'])


def test_write_analysis(tmpdir):
    assert False
