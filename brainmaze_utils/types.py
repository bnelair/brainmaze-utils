# Copyright 2020-present, Mayo Clinic Department of Neurology - Bioelectronics Neurophysiology and Engineering Laboratory
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class ObjDict(dict):
    """
    Dictionary which you can access a) as a dict; b) as a struct with attributes. Can use both foo adding and deleting
    attributes resp items. Inherits from dict

    Missing keys are created on access (auto-vivification), both as items and as
    attributes, so nested structures can be built directly::

        d = ObjDict()
        d.cfg.sub = 3          # {'cfg': {'sub': 3}}
        d['a']['b'] = 1        # {'cfg': ..., 'a': {'b': 1}}

    A consequence (unchanged from v2.0.0): reading a missing public attribute
    creates it as an empty ``ObjDict``, so ``hasattr(d, 'x')`` is always ``True`` for
    public names and creates ``'x'``; test membership with ``'x' in d`` instead.

    Names starting with an underscore (including dunders such as ``__deepcopy__``)
    are never auto-created; reading a missing one raises ``AttributeError``.

    .. note:: **Changed after v2.0.0:**
       Dunder lookups used to auto-create keys too, so :func:`copy.deepcopy`, ``copy``
       and ``pickle`` of an ``ObjDict`` failed (``'ObjDict' object is not callable``)
       and protocol probes (``__array__``, ``_repr_html_``, ...) added junk keys.
    """

    def __init__(self, VT_={}):
        super().__init__(VT_)
        for key in VT_.keys():
            self.__setattr__(key, VT_[key])

    def __setitem__(self, key, value):
        if key in self:
            del self[key]

        super().__setitem__(key, value)
        if not key in self.__dir__():
            self.__setattr__(key, value)

    def __setattr__(self, key, value):
        if key in self.__dir__():
            self.__delattr__(key)

        super().__setattr__(key, value)
        if not key in self:
            self.__setitem__(key, value)

    def __delitem__(self, key):
        value = super().pop(key)
        try:
            super().pop(value, None)
        except: pass
        if key in dir(self):
            self.__delattr__(key)

    def __delattr__(self, key):
        super().__delattr__(key)
        if key in self:
            self.__delitem__(key)

    def __repr__(self):
        return f"{type(self).__name__}({super().__repr__()})"

    def __missing__(self, key):
        self[key] = ObjDict()
        return self[key]

    def __getattr__(self, item):
        # only called when normal attribute lookup fails -> the key does not exist.
        # Public names auto-vivify (v2.0.0 behaviour, d.a.b = 1). Private/dunder names
        # must raise: copy/deepcopy/pickle and other protocols probe them with getattr
        # and break (or get junk keys) if an empty ObjDict is returned.
        if item.startswith('_'):
            raise AttributeError(f"{type(self).__name__!s} has no attribute {item!r}")
        self.__missing__(item)
        return super().__getattribute__(item)

class TwoWayDict(dict):
    def __init__(self, d=None):
        if not isinstance(d, type(None)):
            if isinstance(d, dict):
                for k, v in d.items():
                    self[k] = v

    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        return dict.__len__(self) // 2


def DicToObj(d):
    """
    Converts dictionary into object where, keys - converts keys into attributes
    """
    top = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
        if isinstance(j, dict):
            setattr(top, i, DicToObj(j))
        elif isinstance(j, seqs):
            setattr(top, i,
                    type(j)(DicToObj(sj) if isinstance(sj, dict) else sj for sj in j))
        else:
            setattr(top, i, j)
    return top


def DictToObjDict(d):
    """
    Converts dictionary into object where, keys - converts keys into attributes
    """
    if isinstance(d, dict):
        #print(d.keys())
        top = ObjDict(d)

        #print(top.keys())
        for key in d.keys():
            if isinstance(d[key], dict):
                tmp = DictToObjDict(d[key])
                del top[key]
                top[key] = tmp

        return top
    else: return d


def ObjDictToDict(d):
    """
    Converts dictionary into object where, keys - converts keys into attributes
    """
    if isinstance(d, ObjDict):
        #print(d.keys())
        top = dict(d)

        #print(top.keys())
        for key in d.keys():
            if isinstance(d[key], ObjDict):
                tmp = ObjDictToDict(d[key])
                del top[key]
                top[key] = tmp

        return top
    else: return d

