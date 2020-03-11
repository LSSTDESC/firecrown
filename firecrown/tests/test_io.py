import os
import numpy as np
import pandas as pd

from ..io import write_analysis


def test_write_analysis(tmpdir):
    analysis_id = 'abc'
    output_path = tmpdir
    odir = os.path.join(output_path, 'output_abc')
    arr1 = pd.DataFrame(
        {'g': [0, 1, 2], 'f': [3.0, -1.0, -2.0]}).to_records(index=False)

    write_analysis(analysis_id, output_path, arr1)

    assert os.path.exists(os.path.join(odir, 'analysis.csv'))
    df = pd.read_csv(os.path.join(odir, 'analysis.csv'))
    assert np.array_equal(df['g'], arr1['g'])
    assert np.array_equal(df['f'], arr1['f'])
