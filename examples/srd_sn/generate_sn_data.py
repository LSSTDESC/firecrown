import sacc
from sacc import Sacc
import os
import numpy as np
import pandas as pd
import tarfile
import urllib.request
import datetime
import sys


S = Sacc()


def conversion(h):
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
    h = pd.DataFrame(y1dat)
    h.columns = h.iloc[0, :]
    h = h.iloc[1:, 1:-1]
    h = h.apply(pd.to_numeric)
    h.MU -= 19.36
    h.insert(3, "dz", np.zeros(np.shape(h)[0]))
    row = np.shape(h)[0]
    colu = 13
    join = np.zeros((row, colu))
    hh = pd.DataFrame(np.concatenate([h, join], axis=1))
    hh.columns = col
    h = hh
    h["#name"] = np.linspace(0, np.shape(h)[0] - 1, np.shape(h)[0]).astype(int)
    h = h.T
    return h


if len(sys.argv) == 4:
    path = sys.argv[1]
    HD = sys.argv[2]
    cov = sys.argv[3]
    y1dat = np.loadtxt(open(path + "/" + HD, "rt"), comments="#", dtype=str)
    y1cov = np.loadtxt(path + "/" + cov, comments="#", unpack=True)
    if (y1dat[0][0][0]).isnumeric():
        y1dat = np.array(y1dat).astype(float).T
    else:
        y1dat = conversion(y1dat)
    # print("1. SUCESS")
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
        tf = tarfile.open("LSST_DESC_SRD_v1_release.tar.gz")
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
            tf = tarfile.open("LSST_DESC_SRD_v1_release.tar.gz")
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
            # print("2. SUCESS")


#  set up the sacc data name for the astrophysical sources involved.
sources = ["supernova"]
properties = ["distance"]

# The statistc
statistic = "mu"

# There is no futher specified needed here - everything is scalar.
subtype = None
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
cov = np.zeros((size, size))
count = 1  # since the cov mat starts with the number of lines

S.add_tracer("misc", "sn_ddf_sample")

for i in range(size):

    # Add the appropriate tracer
    S.add_data_point(
        sndata_type, ("sn_ddf_sample",), mb[i], z=zcmb[i]
    )  # can add absmag=-19.9 or other tags
    for j in range(size):
        cov[i, j] = y1cov[count]
        count += 1

    for i in range(size):
        cov[i, i] += (dmb[i]) ** 2

S.add_covariance(cov)
S.metadata["nbin_distmod"] = size
S.metadata["simulation"] = "Y1_DDF_FOUNDATION"
S.metadata["covmat"] = "sys_Y1_DDF_FOUNDATION_0"
S.metadata["creation"] = datetime.datetime.now().isoformat()
S.metadata["info"] = "SN data sets"
S.save_fits("srd-y1-converted.sacc", overwrite=True)

# modify this to have interpolation hubble diagrams
# bias corrections depend on cosmology - systematic vector
