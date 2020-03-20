import multiprocessing

import numpy as np
import emcee
import schwimmbad

from ..loglike import compute_loglike
from ..cosmology import get_ccl_cosmology


def _get_init(pvals, n_walkers, rel_kern=0.1, abs_kern=1e-2):
    n_dims = len(pvals)
    p0 = []
    for i in range(n_walkers):
        rel_fac = (np.random.uniform(size=n_dims)-0.5)*2*rel_kern + 1
        msk = (rel_fac * pvals) == 0
        abs_fac = np.zeros(n_dims)
        abs_fac[msk] = abs_kern
        p0.append(rel_fac * pvals + abs_fac)

    return p0


def _lnprob(p, params, config, data):
    for k, v in zip(params, p):
        data['parameters'][k] = v
    cosmo = get_ccl_cosmology(data['parameters'])
    try:
        loglike, _ = compute_loglike(cosmo=cosmo, data=data)
    except Exception:
        loglike = -np.inf
    return loglike


def run_emcee(config, data, *, parameters, n_steps,
              n_walkers='max(2*n_dims, 20)',
              n_workers=None, backend='serial'):
    """Run the emcee sampler.

    Parameters
    ----------
    config : dict
        The raw config file as a dictionary.
    data : dict
        The result of calling `firecrown.config.parse` on an input YAML
        config.
    parameters : list of str
        A list of the parameters to sample.
    n_steps : int
        The number of MCMC steps to take.
    n_walkers : str or int
        If `n_walkers` is an integer, then emcee uses this many walkers. This
        parameter also accepts strings specifying the number of walkers in
        terms of the number of dimensions, `n_dims` and any of the usual python
        builtins. The default is 'max(2*n_dims, 20)'.
    n_workers : int or None
        The number of workers to use. Only valid with the 'joblib' and
        'multiprocessing' backends, ignored otherwise. The default value of
        `None` will set the numner of workers to `multiprocessing.cpu_count()`.
    backend : str
        The processing backend to use. The options are one of
            'multiprocessing' : python multiprocessing
            'joblib' : joblib lokey-based parallelism (OpenMP safe)
            'mpi' : MPI-base parallelism via mpi4py
            'serial' : use a single process
        The default is 'serial'.

    Returns
    -------
    statistics : dict
        The predicted statistics at the highest likelihood point seen.
    samples : numpy record array, shape (n_walkers * n_steps, n_dims+3)
        The post burn-in MCMC chain.
    """

    _n_workers = n_workers or multiprocessing.cpu_count()

    n_dims = len(parameters)

    if isinstance(n_walkers, str):
        n_walkers = int(eval(n_walkers))

    if backend == 'mpi':
        pool_class = schwimmbad.MPIPool
        pool_kwargs = {}
    elif backend == 'joblib':
        pool_class = schwimmbad.JoblibPool
        pool_kwargs = dict(n_jobs=_n_workers, backend='loky', max_nbytes=None)
    elif backend == 'multiprocessing':
        pool_class = schwimmbad.MultiPool
        pool_kwargs = dict(processes=_n_workers)
    elif backend == 'serial':
        pool_class = schwimmbad.SerialPool
        pool_kwargs = {}
    else:
        raise ValueError("Backend '%s' is not supported!" % backend)

    with pool_class(**pool_kwargs) as pool:
        if (backend == 'mpi' and pool.is_master()) or backend != 'mpi':
            sampler = emcee.EnsembleSampler(
                n_walkers, n_dims, _lnprob,
                args=(parameters, config, data),
                pool=pool)
            pvals = np.array([data['parameters'][p] for p in parameters])
            p0 = _get_init(pvals, n_walkers, rel_kern=0.1, abs_kern=1e-2)

            sampler.run_mcmc(
                p0, n_steps,
                progress=True, skip_initial_state_check=True)
        else:
            sampler = None
            # the only case that gets here is backend == 'mpi' and a worker
            pool.wait()

    if sampler is not None:
        # repackage the chain with iteration and walker numbers
        dtype = [(p, 'f8') for p in parameters]
        dtype += [
            ('emcee_walker', 'i8'),
            ('mcmc_step', 'i8'),
            ('loglike', 'f8')]
        chain = np.zeros(
            sampler.get_chain().shape[0] * sampler.get_chain().shape[1],
            dtype=dtype)
        _chain = sampler.get_chain()
        _log_prob = sampler.get_log_prob()
        loc = 0
        for walker in range(n_walkers):
            for i, p in enumerate(parameters):
                chain[p][loc:loc+n_steps] = _chain[:, walker, i]
            chain['loglike'][loc:loc+n_steps] = _log_prob[:, walker]
            chain['mcmc_step'][loc:loc+n_steps] \
                = np.arange(n_steps, dtype='i8')
            chain['emcee_walker'][loc:loc+n_steps] = walker
            loc += n_steps

        # compute the stats at the best point seen
        max_ind = np.argmax(chain['loglike'])
        for k in parameters:
            data['parameters'][k] = np.asscalar(chain[k][max_ind])
        cosmo = get_ccl_cosmology(data['parameters'])
        _, stats = compute_loglike(cosmo=cosmo, data=data)

        return stats, chain
    else:
        return None, None
