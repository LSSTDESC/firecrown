"""
This class will be replaced by the standard
DESC Parameters class.

This is a placeholder.

"""
from collections import OrderedDict


class ParameterSet(OrderedDict):
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