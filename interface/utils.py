
import numpy as np


def freeze(x):
    """Freezes a dictionary, i.e. makes it immutable and thus hashable."""
    if x is None:
        return None
    if isinstance(x, (list, np.ndarray)):
        return tuple(x)
    elif isinstance(x, dict):
        frozen = {}
        for k, v in sorted(x.items()):
            if isinstance(v, (list, np.ndarray)):
                frozen[k] = tuple(v)
            else:
                frozen[k] = v
        return frozenset(frozen.items())
    raise ValueError('Cannot freeze objects of type {}'.format(type(x)))


def subdict(d, keys=None, nokeys=None):
    """Returns a subdictionary.

    Parameters
    ----------
    d : dict
        A dictionary.
    keys : list or set
        The set of keys to include in the subdictionary. If None use all keys.
    nokeys : list or set
        The set of keys to not include in the subdictionary. If None use no keys.
    """
    keys = set(keys if keys else d.keys())
    nokeys = set(nokeys if nokeys else [])
    return {k: v for k, v in d.items() if k in (keys - nokeys)}

