import os
import sacc

from .parser import (
    _parse_sources,
    _parse_systematics,
    _parse_sn_statistics,
    _parse_likelihood,
)

from ._ccl import compute_loglike, write_stats  # noqa


def parse_config(analysis):
    """Parse a sn analysis.

    Parameters
    ----------
    analysis : dict
        Dictionary containing the SN analysis.

    Returns
    -------
    data : dict
        Dictionary holding all of the data needed for a Nx2pt analysis.
    """
    new_keys = {}
    new_keys["sources"] = _parse_sources(analysis["sources"])
    new_keys["statistics"] = _parse_sn_statistics(analysis["statistics"])
    if "systematics" in analysis:
        print("Not implemented")
        new_keys["systematics"] = _parse_sn_systematics(analysis["systematics"])
    else:
        new_keys["systematics"] = {}
    if "likelihood" in analysis:
        new_keys["likelihood"] = _parse_likelihood(analysis["likelihood"])
        #print('XXXX0  new_keys["likelihood"]= ', new_keys["likelihood"])
    # read data if there is a sacc file
    if "sacc_data" in analysis:
        if isinstance(analysis["sacc_data"], sacc.Sacc):
            sacc_data = analysis["sacc_data"]
        else:
            sacc_data = sacc.Sacc.load_fits(
                os.path.expanduser(os.path.expandvars(analysis["sacc_data"]))
            )

        for src in new_keys["sources"]:
            new_keys["sources"][src].read(sacc_data)
            #print('XXXX0 src = ',src)
        for stat in new_keys["statistics"]:
            #print('XXXX0 stat=',stat)
            new_keys["statistics"][stat].read(sacc_data) #, new_keys["sources"])
        #print('XXXX1 likelihood SN.py')
        
        if "likelihood" in new_keys:
            #print('XXXX2 likelihood SN.py')
            #print(new_keys["sources"])
            #print(new_keys["statistics"])
            #print('XXXX22 likelihood SN.py')
            new_keys["likelihood"].read(
                sacc_data, new_keys["sources"], new_keys["statistics"]
            )
            #print('XXXX3 likelihood SN.py')
        '''
        for like in new_keys["likelihood"]:
            print('XXXX0 likelihood =',like)
            new_keys["likelihood"][like].read(sacc_data, new_keys["sources"], new_keys["statistics"]) #, new_keys["sources"])         
            '''
            
    return new_keys
