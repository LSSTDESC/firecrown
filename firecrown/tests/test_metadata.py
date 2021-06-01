import os
import uuid

import yaml

from ..metadata import write_metadata


def test_write_metadata(tmpdir):
    analysis_id = uuid.uuid4().hex
    odir = os.path.join(tmpdir, "output_%s" % analysis_id)
    os.makedirs(odir, exist_ok=True)

    config_file = os.path.join(tmpdir, "cnf.yaml")
    with open(config_file, "w") as fp:
        fp.write("data: 1\n")

    write_metadata(analysis_id, odir, config_file)

    # we are going light on the tests here
    # let's make sure
    # 1) the output files exist
    # 2) they have some correct data (not checking the whole thing, blah)
    cfg = os.path.join(odir, "config.yaml")
    mtd = os.path.join(odir, "metadata.yaml")
    assert os.path.exists(odir)
    assert os.path.exists(cfg)
    assert os.path.exists(mtd)

    with open(mtd, "r") as fp:
        metadata = yaml.load(fp, Loader=yaml.Loader)
    assert metadata["analysis_id"] == analysis_id

    with open(cfg, "r") as fp:
        config = yaml.load(fp, Loader=yaml.Loader)
    assert config["data"] == 1
