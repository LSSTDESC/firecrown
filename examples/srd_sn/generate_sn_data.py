"""Generate SACC data into file srd-y1-converted.sacc.
"""
from typing import Any, Optional
import os
import tarfile
import urllib.request
import datetime
import sys

import numpy as np
import pandas as pd

from sacc import Sacc
from firecrown.sacc_support import sacc


def conversion(hdg):
    """
    CODE TO CONVERT THE NEW DESC HUBBLE DIAGRAM FILE STRUCTURE
    SIMILAR TO COSMOMC STYLED INPUT HUBBLE DIAGRAM
    FILE STRUCTURE
    """
    col = [
        "#name",
        "zcmb",
        "zhel",
        "dz",
        "mb",
        "dmb",
        "x1",
        "dx1",
        "color",
        "dcolor",
        "3rdvar",
        "d3rdvar",
        "cov_m_s",
        "cov_m_c",
        "cov_s_c",
        "set",
        "ra",
        "dec",
        "biascor",
    ]
    hub = pd.DataFrame(hdg)
    hub.columns = hub.iloc[0, :]
    hub = hub[
        ["ROW", "zCMB", "zHEL", "MU", "MUERR"]
    ]  # takes only "ROW,zCMB,zHEL,MU,MUERR" cols
    hub = hub.iloc[1:, :]
    hub = hub.apply(pd.to_numeric)
    hub.MU -= 19.36
    hub.insert(3, "dz", np.zeros(np.shape(hub)[0]))
    row = np.shape(hub)[0]
    colu = 13
    join = np.zeros((row, colu))
    hhub = pd.DataFrame(np.concatenate([hub, join], axis=1))
    hhub.columns = col
    hub = hhub
    hub["#name"] = np.linspace(0, np.shape(hub)[0] - 1, np.shape(hub)[0]).astype(int)
    hub = np.array(hub.T)
    return hub


if len(sys.argv) == 4:
    path = sys.argv[1]
    HD = sys.argv[2]
    cov = sys.argv[3]
    y1dat = np.loadtxt(f"{path}/{HD}", comments="#", dtype=str)
    y1cov = np.loadtxt(f"{path}/{cov}", comments="#", unpack=True)
    if (y1dat[0][0][0]).isnumeric():
        y1dat = np.array(y1dat).astype(float).T
    else:
        y1dat = conversion(y1dat)
else:
    dirname_year1 = "sndata/Y1_DDF_FOUNDATION"
    dirname_year10 = "sndata/Y10_DDF_WFD_FOUNDATION/"
    url = "https://zenodo.org/record/2662127/files/LSST_DESC_SRD_v1_release.tar.gz?download=1"

    # check if file exists
    if os.path.exists(dirname_year1):
        print("Y1 directory already downloaded")
    else:
        print("Downloading full DESC SRD release files")
        os.mkdir("sndata")
        urllib.request.urlretrieve(url, "sndata/LSST_DESC_SRD_v1_release.tar.gz")
        os.chdir("./sndata/")
        print("Extracting full DESC SRD release files")
        with tarfile.open("LSST_DESC_SRD_v1_release.tar.gz") as tf:
            tf.extractall()
        os.rename(
            "LSST_DESC_SRD_v1_release/forecasting/SN/LikelihoodFiles/Y1_DDF_FOUNDATION/",
            "Y1_DDF_FOUNDATION",
        )
        os.rename(
            "LSST_DESC_SRD_v1_release/forecasting/SN/LikelihoodFiles/Y10_DDF_WFD_FOUNDATION/",
            "Y10_DDF_WFD_FOUNDATION",
        )
        os.chdir("../../")
        print("Done")

        if os.path.exists(dirname_year10):
            print("Y10 directory already downloaded")
        else:
            print("Downloading full DESC SRD release files")
            os.mkdir("sndata")
            urllib.request.urlretrieve(url, "sndata/LSST_DESC_SRD_v1_release.tar.gz")
            os.chdir("./sndata/")
            print("Extracting full DESC SRD release files")
            with tarfile.open("LSST_DESC_SRD_v1_release.tar.gz") as tf:
                tf.extractall()
            os.rename(
                "LSST_DESC_SRD_v1_release/forecasting/SN/LikelihoodFiles/Y1_DDF_FOUNDATION/",
                "Y1_DDF_FOUNDATION",
            )
            os.rename(
                "LSST_DESC_SRD_v1_release/forecasting/SN/LikelihoodFiles/Y10_DDF_WFD_FOUNDATION/",
                "Y10_DDF_WFD_FOUNDATION",
            )
            os.chdir("../../")
            print("Done")

        # read in the Y1 data
    y1dat = np.loadtxt(
        "sndata/Y1_DDF_FOUNDATION/lcparam_Y1_DDF_1.0xFOUNDATION_noScatter.txt",
        unpack=True,
    )
    y1cov = np.loadtxt(
        "sndata/Y1_DDF_FOUNDATION/sys_Y1_DDF_FOUNDATION_0.txt", unpack=True
    )


#  set up the sacc data name for the astrophysical sources involved.
sources = ["supernova"]
properties = ["distance"]

# The statistc
statistic = "mu"

# There is no futher specified needed here - everything is scalar.
subtype: Optional[Any] = None
sndata_type = sacc.build_data_type_name(sources, properties, statistic, subtype)

type_details = sacc.parse_data_type_name(sndata_type)
print(
    "type_details.sources, type_details.properties, type_details.statistic, type_details.subtype"
)
print(
    type_details.sources,
    type_details.properties,
    type_details.statistic,
    type_details.subtype,
)

# Each DataPoint object contains:
# a data type string
# a series of strings listing which tracers apply to it
# a value of the data point
# a dictionary of tags, for example describing binning information


zhel = y1dat[2]  # redshift
zcmb = y1dat[1]  # redshift
mb = y1dat[4]
dmb = y1dat[5]
zmu = np.vstack((zcmb, mb))
size = int(y1cov[0])  # reading the size of the matrix from the first entry
covmat = np.zeros((size, size), dtype=float)
count = 1  # since the cov mat starts with the number of lines
out_name = "srd-y1-converted"

sacc_data = Sacc()
sacc_data.add_tracer("misc", "sn_ddf_sample")

for i in range(size):
    # Add the appropriate tracer
    sacc_data.add_data_point(
        sndata_type, ("sn_ddf_sample",), mb[i], z=zcmb[i]
    )  # can add absmag=-19.9 or other tags
    for j in range(size):
        covmat[i, j] = y1cov[count]
        count += 1
    for i in range(size):
        covmat[i, i] += (dmb[i]) ** 2

sacc_data.add_covariance(covmat)
sacc_data.metadata["nbin_distmod"] = size
sacc_data.metadata["simulation"] = "Y1_DDF_FOUNDATION"
sacc_data.metadata["covmat"] = "sys_Y1_DDF_FOUNDATION_0"
sacc_data.metadata["creation"] = datetime.datetime.now().isoformat()
sacc_data.metadata["info"] = "SN data sets"
sacc_data.save_fits(out_name + ".sacc", overwrite=True)
# modify this to have interpolation hubble diagrams
# bias corrections depend on cosmology - systematic vector
