import os

from ..io import write_analysis


def test_write_analysis(tmpdir):
    analysis_id = 'abc'
    output_path = tmpdir
    odir = os.path.join(output_path, 'output_abc')
    arr1 = "blah"

    write_analysis(analysis_id, output_path, arr1)

    assert os.path.exists(os.path.join(odir, 'chain.txt'))
    with open(os.path.join(odir, 'chain.txt'), "r") as fp:
        assert fp.read().strip() == "blah"
