from __future__ import print_function
import numpy as np
import copy
import sys
from collections import OrderedDict

#
# ParameterSet class
#

class ParameterSet(OrderedDict):
    """A class for holding the names and current values of the parameters that will be varied.

    Typically, these are the parameters that are stepped over in a Markov chain or similar
    code.  This class is little more than a glorified Python dict with a bit of extra
    syntactic sugar to let parameters be accessed as attributes rather than keys.

    A ParameterSet object is constructed with arbitrary kwargs, giving the names of the parameters
    to include.  The value of each should be either a type (float, int or bool) or a value with
    one of these types.

    Example:

        >>> cosmo_params = ParameterSet(Omega_m=float, h=float, Lambda=float, As=float, ns=float,
        ...                             sum_neutrino_masses=float, n_neutrino_species=int,
        ...                             inverted_neutrino_hierarchy=bool)
        >>> ia_params = ParameterSet(a1=1.4, a2=2.3)

    In addition, the value of an argument may be another ParameterSet object, in which case the
    parameters will be accessed hierarchically (not unlike a dict inside another dict):

        >>> bias_params = ParameterSet(b1=3.0, b2=0.0)
        >>> sys_params = ParameterSet(ia=ia_params, bias=bias_params)
        >>> all_params = ParameterSet(cosmo=cosmo_params, sys=sys_params)

    You can set or access hierarchical parameter values with the same syntax as a dict or with
    a single key string using a '.' to separate the keys in the different levels:

        >>> sys_params['ia']['a1'] = 1.2
        >>> sys_params['bias.b1'] = 3.

    Finally, you can access the parameters as attributes instead:

        >>> all_params.sys.bias.b2 = 0.2
        >>> all_params.cosmo.sum_neutrino_masses = 0.6

    """
    _valid_types = [ float, int, bool ]

    def __init__(self, **kwargs):
        super(ParameterSet,self).__init__()
        for key, value in kwargs.items():
            if isinstance(value, ParameterSet):
                OrderedDict.__setitem__(self, key, value)
            elif isinstance(value, type):
                if value not in self._valid_types:
                    raise ValueError("Type %s not supported for parameter %s"%(value, key))
                OrderedDict.__setitem__(self, key, value(0))
            else:
                value = self._check_type(key, value, self._valid_types)
                OrderedDict.__setitem__(self, key, value)

    def _check_type(self, key, value, valid_types):
        # If value has an invalid type, see if it is convertible into either int or float.
        if type(value) not in valid_types:
            # If not one of these, see if it is implicitly convertible to int or float.
            try:
                if int in valid_types and int(value) == value: return int(value)
                if float in valid_types and float(value) == value: return float(value)
            except (TypeError, ValueError):
                pass
        if type(value) not in self._valid_types:
            raise ValueError("Type %s not supported for parameter %s"%(type(value), key))
        return value

    def __getitem__(self, key):
        if key in self:
            return OrderedDict.__getitem__(self, key)
        elif '.' in key:
            k1, k2 = key.split('.',1)
            if k1 in self:
                params = OrderedDict.__getitem__(self, k1)
                if  isinstance(params, ParameterSet):
                    return params[k2]
        # If didn't return somewhere above, then key is invalid.
        raise KeyError("Key %s is invalid"%key)

    def __setitem__(self, key, value):
        if key in self:
            value_type = type(OrderedDict.__getitem__(self, key))
            value = self._check_type(key, value, [value_type])
            if type(value) != value_type:
                raise ValueError("When setting %s, expected a %s.  Got %s"%(key, value_type, value))
            OrderedDict.__setitem__(self, key, value)
            return
        elif '.' in key:
            k1, k2 = key.split('.',1)
            if k1 in self:
                params = OrderedDict.__getitem__(self, k1)
                if  isinstance(params, ParameterSet):
                    params[k2] = value
                    return
        # If didn't return somewhere above, then key is invalid.
        raise KeyError("Key %s is invalid"%key)

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError("Attribute %s is invalid"%key)

    def __setattr__(self, key, value):
        try:
            self.__setitem__(key, value)
        except KeyError:
            if key[0] == '_':
                OrderedDict.__setattr__(self, key, value)
            else:
                raise AttributeError("Attribute %s is invalid"%key)

    def copy(self):
        ret = ParameterSet.__new__(ParameterSet)
        super(ParameterSet,ret).__init__()
        for key, value in self.items():
            if isinstance(value, ParameterSet):
                OrderedDict.__setitem__(ret, key, value.copy())
            else:
                OrderedDict.__setitem__(ret, key, value)
        return ret

    def full_keys(self):
        """Return a complete list of all keys, including nested parameters.
        """
        keys = []
        for key, value in self.items():
            if isinstance(value, ParameterSet):
                suffixes = value.full_keys()
                keys += [ key + '.' + s for s in suffixes ]
            else:
                keys.append(key)
        return keys

    # The str representation isn't valid syntax, since the keys may have dots in them,
    # but is is more compact and probably more readable in most cases.
    def __str__(self):
        s = 'ParameterSet('
        for i,key in enumerate(self.full_keys()):
            if i > 0: s += ', '
            s += '%s=%r'%(key, self[key])
        s += ')'
        return s

    # The repr should be eval-able to return something equivalent to self.
    def __repr__(self):
        s = 'ParameterSet('
        for i,(key, value) in enumerate(self.items()):
            if i > 0: s += ', '
            s += '%s=%r'%(key, value)
        s += ')'
        return s

    def __eq__(self, other):
        if not isinstance(other, ParameterSet): return False
        if set(self.keys()) != set(other.keys()): return False
        for k in self:
            if self[k] != other[k]: return False
        return True


class TheoryVector(object):
    """A class that can generate a theory vector.
    
    Inputs are an input cosmology, some sources, some target statistics to include, and 
    possibly some systematics.  Most systematics are attached to the sources directly,
    but some may be passed here as more global or cosmological systematics.
    """
    def __init__(self, cosmology, sources, statistics, systematics=None):
        self.cosmology = cosmology
        self.sources = sources
        self.statistics = statistics

        # Build our complete list of systematics
        # First, any systematics given directly here are global-level systematics, not tied
        # to any particular source. E.g. baryon effects.
        if systematics is None:
            self.systematics = []
        else:
            self.systematics = systematics

        # Add in any systematics that the sources define
        for source in self.sources:
            self.systematics += source.systematics

        self.validate()

    def validate(self):
        """Validate the inputs to make sure everything in consistent.
        """
        # Check that our sources have unique names.  Maybe other sanity checks about the sources.
        names = [source.name for source in self.sources]
        if len(names) != len(set(names)):
            raise ValueError("Some sources have identical names!")
        # Other sanity checks? ... 

        # Check that we have the appropriate sources to do each statistic, that the cosmology
        # is valid for this statistic, maybe other sanity checks.
        for stat in self.statistics:
            stat.validate(self.cosmology, self.sources)

    def get_params(self):
        """Build a parameter vector.
        
        These are the values that need to be chained over, including both cosmological parameters
        of interest, and nuisance parameters.

        Mostly the calling routine probably just needs the length of this vector
        """
        # Start with the variable cosmological parameters
        params = self.cosmology.get_params()

        self._param_index = [0] * (len(self.systematics)+1)
        self._param_index[0] = len(params)

        # Add in nuisance parameters
        for i_sys, sys in enumerate(self.systematics):
            sys_params = sys.get_nuisance_params(self.sources)
            params += sys_params
            self._param_index[i_sys+1] = len(params)

        return params


    def build_vector(self, params):
        """Build up a vector for a given step in a chain
        """
        # Get a new cosmology for this step in the chain.
        cosmology = self.cosmology.with_params(params[0:self._param_index[0]])

        systematics = []
        for i_sys, sys in enumerate(self.systematics):
            sys_params = params[self._param_index[i_sys]:self._param_index[i_sys+1]]
            systematics.append(sys.with_params(sys_params))

        # Presumably do any initial calculations that are needed.  Pdelta, etc.?
        cosmology.initialize()

        # Note: some systamtics might be in more than one of these catagories.
        input_sys = [ sys for sys in self.systematics if sys.affects_input() ]
        output_sys = [ sys for sys in self.systematics if sys.affects_output() ]
        calc_sys = [ sys for sys in self.systematics if sys.requires_calculation() ]

        for sys in input_sys:
            # Not exactly sure if I understood the meaning of input systematics, but maybe
            # something like this to adjust the nominal Pdelta, etc.
            cosmology = sys.adjust_input(cosmology)

        # Build up the statistcs vectors
        stat_vectors = []
        for stat in self.statistics:
            # I guess the calculation statistics need to be passed into this function.
            calc_sys = [ sys for sys in calc_sys if sys.is_relevant_to(stat) ]
            v = stat.build_vector(cosmology, self.sources, calc_sys)

            # Adjust the output vectors as needed
            for sys in output_sys:
                sys.adjust_output(v, stat, cosmology)

        # Combine into a single theory vector
        return np.concatenate(stat_vectors)



# Some test functions
def test_params():
    """Basic tests of the ParameterSet functionality
    """
    # Use the examples in the doc string to make sure they run correctly.
    cosmo_params = ParameterSet(Omega_m=float, h=float, Lambda=float, As=float, ns=float,
                                sum_neutrino_masses=float, n_neutrino_species=int,
                                inverted_neutrino_hierarchy=bool)

    print('cosmo_params = ',cosmo_params)
    assert cosmo_params['Omega_m'] == 0.
    assert type(cosmo_params['Omega_m']) is float
    assert cosmo_params['n_neutrino_species'] == 0
    assert type(cosmo_params['n_neutrino_species']) is int
    assert cosmo_params['inverted_neutrino_hierarchy'] == False
    assert type(cosmo_params['inverted_neutrino_hierarchy']) is bool

    ref_list = ['Omega_m', 'h', 'Lambda', 'As', 'ns', 'sum_neutrino_masses',
                'n_neutrino_species', 'inverted_neutrino_hierarchy']
    if sys.version_info >= (3,6):
        # Starting in python 3.6, kwargs are kept in order.  Prior to this, the order is arbitrary.
        # cf. PEP 468
        np.testing.assert_equal(cosmo_params.keys(), ref_list)
        np.testing.assert_equal(cosmo_params.full_keys(), ref_list)
    else:
        np.testing.assert_equal(set(cosmo_params.keys()), set(ref_list))
        np.testing.assert_equal(set(cosmo_params.full_keys()), set(ref_list))

    print('repr(cosmo_params) = ',repr(cosmo_params))
    assert cosmo_params == eval(repr(cosmo_params))
    assert cosmo_params == eval(str(cosmo_params))  # No hierachy, so this one also works.

    # Check construction with values.
    ia_params = ParameterSet(a1=1.4, a2=2.3)
    bias_params = ParameterSet(b1=3.0, b2=0.0)
    assert ia_params.a1 == ia_params['a1'] == 1.4
    assert ia_params.a2 == ia_params['a2'] == 2.3
    assert bias_params.b1 == bias_params['b1'] == 3.0
    assert bias_params.b2 == bias_params['b2'] == 0.0
    assert ia_params == eval(repr(ia_params))
    assert ia_params == eval(str(ia_params))

    # Check construction with numpy values.
    bias_params2 = ParameterSet(b1=np.float32(3.0), b2=np.float64(0.0))
    assert bias_params.b1 == bias_params['b1'] == 3.0
    assert bias_params.b2 == bias_params['b2'] == 0.0
    assert type(bias_params.b1) is float  # Not float32
    assert type(bias_params.b2) is float  # Not float64
    nu_params = ParameterSet(sum_neutrino_masses=np.float32(0.5), n_neutrino_species=np.int16(3),
                             inverted_neutrino_hierarchy=np.bool(True))
    assert type(nu_params.sum_neutrino_masses) is float
    assert type(nu_params.n_neutrino_species) is int
    assert type(nu_params.inverted_neutrino_hierarchy) is bool
    assert nu_params.sum_neutrino_masses == 0.5
    assert nu_params.n_neutrino_species == 3
    assert nu_params.inverted_neutrino_hierarchy == True
    assert nu_params == eval(repr(nu_params))
    assert nu_params == eval(str(nu_params))

    # Check hierarchical construction.
    cosmo_params = ParameterSet(Omega_m=0.3, h=0.7, Lambda=0.7, sigma_8=0.72, nu=nu_params)
    sys_params = ParameterSet(ia=ia_params, bias=bias_params)
    all_params = ParameterSet(cosmo=cosmo_params, sys=sys_params)
    print('all_params = ',all_params)
    print('repr = ',repr(all_params))
    assert cosmo_params == eval(repr(cosmo_params))
    assert sys_params == eval(repr(sys_params))
    assert all_params == eval(repr(all_params))
    assert cosmo_params != None
    assert cosmo_params != sys_params
    assert all_params != sys_params
    all_params2 = all_params.copy()
    assert all_params2 == all_params
    all_params2.sys.ia.a1 = 10
    assert all_params2 != all_params

    assert sys_params['ia']['a1'] == 1.4
    assert sys_params['bias.b1'] == 3.
    assert all_params.sys.ia.a2 == 2.3
    assert all_params.cosmo['nu'].sum_neutrino_masses == 0.5

    ref_list = ['cosmo.Omega_m', 'cosmo.h', 'cosmo.Lambda', 'cosmo.sigma_8',
                'cosmo.nu.sum_neutrino_masses', 'cosmo.nu.n_neutrino_species',
                'cosmo.nu.inverted_neutrino_hierarchy',
                'sys.ia.a1', 'sys.ia.a2', 'sys.bias.b1', 'sys.bias.b2']
    assert set(all_params.keys()) == set(['cosmo', 'sys'])
    assert set(all_params.cosmo.keys()) == set(['Omega_m', 'h', 'Lambda', 'sigma_8', 'nu'])
    assert set(sys_params.keys()) == set(['ia', 'bias'])
    assert set(all_params.full_keys()) == set(ref_list)

    # Finally, check some invalid constructions and calls
    np.testing.assert_raises(ValueError, ParameterSet, blah=str)
    np.testing.assert_raises(ValueError, ParameterSet, blah='string')
    np.testing.assert_raises(KeyError, all_params.__getitem__, 'blah')
    np.testing.assert_raises(TypeError, all_params.__getitem__, 3)
    np.testing.assert_raises(KeyError, all_params.__setitem__, 'blah', 3)
    np.testing.assert_raises(TypeError, all_params.__setitem__, 3, 3)
    np.testing.assert_raises(ValueError, all_params.__setitem__, 'sys', 3)
    np.testing.assert_raises(ValueError, all_params.__setitem__, 'sys.bias.b1', 3 + 1j)
    np.testing.assert_raises(ValueError, all_params.__setitem__, 'sys.bias.b1', '3')
    np.testing.assert_raises(ValueError, all_params.__setitem__, 'sys.bias.b1', 'three')
    np.testing.assert_raises(AttributeError, all_params.__getattr__, 'blah')
    np.testing.assert_raises(AttributeError, all_params.__setattr__, 'blah', 5)
    np.testing.assert_raises(TypeError, all_params.__getattr__, 3)


if __name__ == '__main__':
    test_params()
