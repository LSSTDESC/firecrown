import numpy as np
from firecrown.models.cluster_mass import ClusterMass, ClusterMassArgument
from firecrown.models.cluster_mass_rich_proxy import ClusterMassRich, ClusterMassRichBinArgument
import pytest

from typing import Any, Dict
import itertools
import math


import pyccl as ccl


def cluster_init():
    pivot_redshift=0.6
    pivot_mass = 14.625862906
    log_pivot_mass = pivot_mass * np.log(10.0)
    log1p_pivot_redshift = np.log1p(pivot_redshift)
    
    z=1
    logM=15
    lnM = logM * np.log(10)
    

    cluster_mass_rich = ClusterMassRich(pivot_mass, pivot_redshift)
    
    cluster_mass_rich.mu_p0 = 3.00
    cluster_mass_rich.mu_p1 = .86
    cluster_mass_rich.mu_p2 =0
    cluster_mass_rich.sigma_p0 = 3.0
    cluster_mass_rich.sigma_p1 = .7
    cluster_mass_rich.sigma_p2 = 0.0
    
    return cluster_mass_rich
    
    
@pytest.fixture(name="richness_args")
def fixture_cluster_richness_args():
    """Richness and sd for a given mass."""
    cluster_obj = cluster_init()
    logM=15
    z=1
    
    
    richness_args=cluster_obj.cluster_mass_lnM_obs_mu_sigma(logM, z)
    
    assert(abs(richness_args[0]-3.74)<.1)
    assert(abs(richness_args[1]-3.60)<.1)
    
    return richness_args


##Testing ClusterMassRichBinArgument

@pytest.fixture(name="probability")
def fixture_cluster_probability():
    """Probability of a particular richness bin."""
    
    cluster_obj = cluster_init()
    
    
    cluser_mass_rich_bin_argument=ClusterMassRichBinArgument(cluster_obj,13.0, 17.0, 1, 100)
    
    z=1
    logM=15
    
    probability=cluser_mass_rich_bin_argument.p(logM,z)
    assert abs(probability-0.655<0.1)
    
    return probability
    


@pytest.fixture(name="richness_bins")
def fixture_cluster_richness_bins():
    """Returns binned array of richness."""
    logM=15
    z=1
    cluster_obj = cluster_init()
    rich_bin_test=np.array([20,40,60,80])
    
    
    cluster_mass_rich_bin_argument=ClusterMassRichBinArgument(cluster_obj,13.0, 17.0, 1, 100)
    
    richness_bins =cluster_obj.gen_bins_by_array(rich_bin_test)
    
    assert richness_bins[0].logM_obs_lower==rich_bin_test[0]
    assert richness_bins[-1].logM_obs_upper==rich_bin_test[-1]
    
    for i in range(len(rich_bin_test)-1):
        return(richness_bins[i], richness_bins[i+1])
    
    
    
    
def test_cluster_richness_bins(richness_bins):
    pass

def test_cluster_richness_args(richness_args):
    pass

def test_cluster_probability(probability):
    pass
    




    
