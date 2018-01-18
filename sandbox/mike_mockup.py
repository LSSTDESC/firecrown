from __future__ import print_function
import numpy as np
import copy
import sys
from collections import OrderedDict

#
# ParameterSet class keeps track of the variable parameters being chained over.
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


#
# TheoryVector is the main driver class that generates a theory vector for each step in a chain.
#

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
        self.systematics = systematics

        # Make a dict that connects names to source objects
        self.source_by_name = { s.name : s for s in sources }

        self.validate()

    def validate(self):
        """Validate the inputs to make sure everything in consistent.
        """
        # Check that our sources have unique names.  Maybe other sanity checks about the sources.
        names = [ s.name for s in self.sources ]
        if len(names) != len(set(names)):
            raise ValueError("Some sources have identical names!")
        # Other sanity checks? ...

        # Check that we have the appropriate sources to do each statistic, that the cosmology
        # is valid for this statistic, maybe other sanity checks.
        self.stat_sources = {}
        for stat in self.statistics:
            for source_name in stat.source_names:
                if source_name not in self.source_by_name:
                    raise ValueError("Source %s needed by %s not found"%(source_name, stat.name))

            stat_sources = [ self.source_by_name[n] for n in stat.source_names ]
            self.stat_sources[stat.name] = stat_sources

            # Any other checks that the statistic might want to do.
            stat.validate(self.cosmology, stat_sources)

        # Check that the systematics have whatever they need
        for source in self.sources:
            for sys in source.systematics:
                sys.validate(source)


    def get_params(self):
        """Build a ParameterSet for the cosmology and all systematics.

        These are the values that need to be chained over, including both cosmological parameters
        of interest, and nuisance parameters.
        """
        # Start with the variable cosmological parameters
        kwargs = { 'cosmo' : self.cosmology.get_variable_params() }

        # Add in nuisance parameters
        for sys in self.systematics:
            sys_params = sys.get_nuisance_params(self.cosmology)
            if sys_params is not None:
                kwargs[sys.name] = sys_params

        for source in self.sources:
            source_kwargs = {}
            for sys in source.systematics:
                sys_params = sys.get_nuisance_params(self.cosmology, source)
                if sys_params is not None:
                    source_kwargs[sys.name] = sys_params
            kwargs[source.name] = ParameterSet(**source_kwargs)

        return ParameterSet(**kwargs)

    def build_vector(self, params):
        """Build up a vector for a given step in a chain
        """

        # Note: Not sure how best to get the covariance calculation in here.  It could either
        #       be returned as well from this method, where each statistic would know how to
        #       compute its own covariance and then something be done for the cross-statistics
        #       blocks.  Or there could be a parallel method that looks a lot like this in terms
        #       of the structure of hot the cosmology, sources get adjusted, but then there would
        #       be a different method at the end for adjusting the covariance rather than the
        #       vector.
        #       The former might be better if the the covariance is different for each step in the
        #       chain, but the latter would be better if the covariance is kept fixed.  (And having
        #       them be two functions would still work even if the covariance is changing each time,
        #       since it would probably not be much overhead to run both functions for each step.)

        # Get a new cosmology for this step in the chain.
        cosmology = self.cosmology.with_params(params.cosmo)

        # Update the systematics with the parameters for this step.
        global_systematics = []
        for sys in self.systematics:
            if sys.name in params:
                sys = sys.with_params(params[sys.name])
            global_systematics.append(sys)

        source_systematics = {}
        for source in self.sources:
            source_systematics[source.name] = []
            for sys in source.systematics:
                if sys.name in params[source.name]:
                    sys = sys.with_params(params[source.name][sys.name])
                source_systematics[source.name].append(sys)

        # Presumably do any initial calculations that are needed.  Pdelta, etc.?
        cosmology.initialize()

        # Update the cosmology as appropriate.
        for sys in self.systematics:
            cosmology = sys.adjust_cosmology(cosmology)

        # Update the sources as appropriate.
        sources = []
        for source in self.sources:
            for sys in source_systematics[source.name]:
                source = sys.adjust_source(source, cosmology)
            sources.append(source)

        # Build up the statistcs vectors
        stat_vectors = []
        for stat in self.statistics:
            stat_sources = self.stat_sources[stat.name]
            # Get the raw theory vector for this statistics
            v = stat.build_vector(cosmology, stat_sources)

            # Update this vector as appropriate
            for sys in global_systematics:
                v = sys.adjust_vector(v, stat, stat_sources, cosmology)

            for source in stat_sources:
                for sys in source_systematics[source.name]:
                    v = sys.adjust_vector(v, stat, stat_sources, cosmology)

            stat_vectors.append(v)

        # Combine into a single theory vector
        return np.concatenate(stat_vectors)


#
# Systematic classes modify various aspects of the calculation appropriately.
# Included here are three toy systematics models that apply to different stages of the process.
#

class Systematic(object):
    """Base class for the various systematic types.
    """
    def __init__(self):
        pass

    def validate(self, source):
        """Check that the source has the required metadata to use this systematic.
        """
        pass

    def get_nuisance_params(self, cosmology, source=None):
        """Get the appropriate nuisance parameters for this systematic

        The base class implementation is to have no nuisance parameters.
        Subclasses that do have nuisance parameters will override this and return a
        ParameterSet instance.
        """
        return None

    def adjust_cosmology(self, cosmology):
        """Adjust the cosmology appropriately according to the current systematic.

        The base class implementation is a no op.

        Subclasses that need to do something here will override this functionality,
        returning a new copy of the cosmology with appropriate modifications.
        """
        return cosmology

    def adjust_source(self, source, cosmology):
        """Adjust source information appropriately according to the current systematic.

        The base class implementation is a no op.

        Subclasses that need to do something here will override this functionality,
        returning a new copy of the source with appropriate modifications.
        """
        return source

    def adjust_vector(self, vector, stat, sources, cosmology):
        """Adjust the theory vector from a given statistic appropriately.

        The base class implementation is a no op.

        Unlike other adjust_* methods, there is probably no need to make a copy of the
        vector.  So the vector may be modified in place.
        """
        return vector

class BaryonEffects(Systematic):
    """Systematic implementing the effects of baryons on the cosmological power spectrum.
    """
    name = 'BaryonEffects'

    def __init__(self, kmin=1000., amplitude=0.):
        self.kmin = kmin
        self.amp = amplitude

    def get_nuisance_params(self, cosmology, source=None):
        return ParameterSet(amplitude=float)

    def with_params(self, params):
        return BaryonEffects(self.kmin, params.amplitude)

    def adjust_cosmology(self, cosmology):
        new_cosmo = copy.copy(cosmology)
        # Should do something here presumably...
        return new_cosmo

class NonlinearBias(Systematic):
    """Systematic implementing the effects of non-linear bias.
    """
    name = 'NonlinearBias'

    def __init__(self, b1=1., b2=0., fixed=False):
        self.b1 = b1
        self.b2 = b2
        self.fixed = fixed

    def get_nuisance_params(self, cosmology, source):
        # N.B. source is a require parameter here, since a NonlinearBias must be attached to a
        # particular source object.
        if self.fixed:
            return None
        else:
            return ParameterSet(b1=float, b2=float)

    def with_params(self, params):
        # This has a cheap constructor, so go ahead and use that.
        return NonlinearBias(params.b1, params.b2)

    def adjust_source(self, source, cosmology):
        new_source = copy.copy(source)
        # Should do something here presumably...
        return new_source

class SimpleIA(Systematic):
    """Systematic implementing some model of intrinsic alignments
    """
    name = 'SimpleIA'

    def __init__(self, a1=0., a2=0.):
        self.a1 = a1
        self.a2 = a2

    def validate(self, source):
        # For this toy model, we posit that the IA model is invalid above z=0.6.
        if source.mean_z > 0.6:
            raise ValueError("SimpleIA is invalid above redshift 0.6")

    def get_nuisance_params(self, cosmology, source):
        return ParameterSet(a1=float, a2=float)

    def with_params(self, params):
        return SimpleIA(params.a1, params.a2)

    def adjust_vector(self, v, stat, sources, cosmology):
        # Should do something here presumably...
        return v


# Note: Everything below here is very much a toy model just to write the tests.
#       These aren't trying to be complete in any sense.
class Cosmology(object):
    """A toy class representing cosmological information.
    """
    def __init__(self, fixed_params, variable_params):
        # This is probably not the interface you'd want for this, but it's fine for setting up
        # the tests below.
        self.fixed_params = fixed_params
        self.variable_params = variable_params

    def get_variable_params(self):
        return self.variable_params

    def initialize(self):
        pass

    def with_params(self, params):
        return Cosmology(self.fixed_params, params)


class Statistic(object):
    """Base class for the various statistic types.
    """
    def __init__(self):
        pass

    def validate(self, cosmology, sources):
        pass

    def build_vector(self, cosmology, sources):
        raise NotImplementedError("Each derived class needs to implement build_vector")

class ShearShearCorrelation(Statistic):
    """A toy two-point shear-shear correlation function.
    """
    def __init__(self, name, source_names, theta_min=10, theta_max=200, nbins=5):
        self.name = name
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.nbins = nbins
        self.source_names = source_names

    def validate(self, cosmology, sources):
        pass

    def build_vector(self, cosmology, sources):
        return np.zeros(self.nbins)

class Source(object):
    """Base class for a source type
    """
    def __init__(self):
        pass

class WLShear(Source):
    """A toy class representing the data in a WL shear catalog
    """
    def __init__(self, name, mean_z):
        self.name = name
        self.mean_z = mean_z



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


def test_theory_vector():
    """Basic tests of the TheoryVector usage
    """
    # We'll model a very simple use case of shear-shear correlations with 3 tomo bins.

    # Cosmology has 2 variable parameters: Omega_m and sigma_8.
    cosmology = Cosmology(fixed_params=None,
                          variable_params=ParameterSet(Omega_m=float, sigma_8=float))

    # The sources are shears in 3 bins.  These don't actually have any information except the name.
    ntomo = 3
    mean_z = [0.31, 0.57, 0.88]
    sources = [ WLShear(name='shears_%d'%tomo, mean_z=mean_z[tomo]) for tomo in range(ntomo) ]
    assert len(sources) == ntomo

    # There are 6 two-point correlations, 3 auto and 3 cross.
    nbins = 5
    stats = [ ShearShearCorrelation(name='shear2pt_%d_%d'%(i,j),
                                    source_names=['shears_%d'%i, 'shears_%d'%j],
                                    nbins=nbins)
              for i in range(ntomo) for j in range(i,ntomo) ]
    assert len(stats) == ntomo * (ntomo+1) / 2

    # We have three systematics:
    # BaryonEffects modifies some stuff about the cosmology
    # NonlinearBias modifies the bias term for each source
    # SimpleIA modifies the measured two-point correlation function
    # The latter two are attached to sources.

    for source in sources:
        # Let's say we only apply IA for z < 0.6, so the first 2 tomo bins.
        # And the bias is fixed for z < 0.4, so only variable in the higher 2 tomo bins.
        bias = NonlinearBias(fixed=source.mean_z < 0.4)
        if source.mean_z < 0.6:
            ia = SimpleIA()
            source.systematics = [bias, ia]
        else:
            source.systematics = [bias]

    baryons = BaryonEffects(kmin=1000)  # Presumably some minimum k at which baryons affect Pdelta.
    cosmic_shear = TheoryVector(cosmology, sources, stats, [baryons])

    # Get the set of parameters including nuisance parameters
    params = cosmic_shear.get_params()
    print('params = ',params.full_keys())
    # There are 2 cosmo params, 1 baryon, 2*2 bias, and 2*2 ia.
    assert len(params.full_keys()) == 11

    # Simulate a step in a MCMC with specific values for each parameter.
    # Note: not sure where limits and priors fit in.  They might belong in the ParameterSet
    # class as further information about each parameter.  Or possibly in a separate structure.
    # For now, I ignore that, and just draw random values from 0-1 for each parameter.
    for key in params.full_keys():
        params[key] = np.random.random()
    print('params for this step = ',params)
    vector = cosmic_shear.build_vector(params)

    print('vector has %d elements.'%len(vector))
    assert len(vector) == nbins * len(stats)

def test_invalid():
    """Test some invalid operations with TheoryVector
    """
    cosmology = Cosmology(fixed_params=None,
                          variable_params=ParameterSet(Omega_m=float, sigma_8=float))
    source1 = WLShear(name='shears_0', mean_z=0.5)
    source2 = WLShear(name='shears_1', mean_z=0.8)

    stat1 = ShearShearCorrelation(name='shear2pt', source_names=['shears_0'])
    stat2 = ShearShearCorrelation(name='shear2pt', source_names=['shears_1'])
    stat3 = ShearShearCorrelation(name='shear2pt', source_names=['shears_1', 'shears_2'])

    # Multiple sources with the same name is not allowed.
    np.testing.assert_raises(ValueError, TheoryVector, cosmology, [source1, source1], [stat1])

    # Stats needing a source not listed
    np.testing.assert_raises(ValueError, TheoryVector, cosmology, [source1], [stat2])


def test_config():
    """Demo of what a config-based version of test_theory_vector might look like:

    Note: A real implementation would do a lot more checking of the validity of the input.
          I don't do any such checking here.
    """
    import yaml
    with open('mike_demo.yaml') as fin:
        config = yaml.load(fin)

    # Construct the Cosmology instance
    cosmology = Cosmology(fixed_params=ParameterSet(**(config['cosmology']['fixed'])),
                          variable_params=ParameterSet(**(config['cosmology']['variable'])))

    # Construct the Sources, including their associated Systematics
    sources = []
    for source_dict in config['sources']:
        source_type = eval(source_dict.pop('type'))
        sys_config = source_dict.pop('systematics', [])  # These are the config specs
        systematics = [] # These will be the constructed Systematics instances
        for sys_dict in sys_config:
            sys_type = eval(sys_dict.pop('type'))
            systematics.append(sys_type(**sys_dict))
        source = source_type(**source_dict)
        source.systematics = systematics
        sources.append(source)
    assert len(sources) == 3

    # Construct the Statistics
    stats = []
    for stat_dict in config['statistics']:
        stat_type = eval(stat_dict.pop('type'))
        stats.append(stat_type(**stat_dict))
    assert len(stats) == 6

    # Construct the global Systematics
    systematics = []
    for sys_dict in config.get('systematics',[]):
        sys_type = eval(sys_dict.pop('type'))
        systematics.append(sys_type(**sys_dict))

    theory_vector = TheoryVector(cosmology, sources, stats, systematics)

    # Run through the same usage tests we did in test_theory_vector()
    params = theory_vector.get_params()
    assert len(params.full_keys()) == 11
    for key in params.full_keys():
        params[key] = np.random.random()
    vector = theory_vector.build_vector(params)
    assert len(vector) == 30


if __name__ == '__main__':
    test_params()
    test_theory_vector()
    test_invalid()
    test_config()
