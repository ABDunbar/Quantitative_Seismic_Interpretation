# -*- coding: utf-8 -*-
"""
Smoothers.

:copyright: 2015 Agile Geoscience
:license: Apache 2.0
"""
import numpy as np
import scipy.ndimage

from bruges.bruges import BrugesError
from bruges.util import nearest
from bruges.util import rms as rms_

# TODO:
#     - 1D and 2D Gaussian (or, better, n-D)
#     - See how these handle Nans, consider removing, interpolating, replacing.


def mean(arr, size=5):
    """
    A linear n-D smoothing filter. Can be used as a moving average on 1D data.

    Args:
        arr (ndarray): an n-dimensional array, such as a seismic horizon.
        size (int): the kernel size, e.g. 5 for 5x5. Should be odd,
            rounded up if not.

    Returns:
        ndarray: the resulting smoothed array.
    """
    arr = np.array(arr, dtype=np.float)

    if not size // 2:
        size += 1

    return scipy.ndimage.generic_filter(arr, np.mean, size=size)


def rms(arr, size=5):
    """
    A linear n-D smoothing filter. Can be used as a moving average on 1D data.

    Args:
        arr (ndarray): an n-dimensional array, such as a seismic horizon.
        size (int): the kernel size, e.g. 5 for 5x5. Should be odd,
            rounded up if not.

    Returns:
        ndarray: the resulting smoothed array.
    """
    arr = np.array(arr, dtype=np.float)

    if not size // 2:
        size += 1

    return scipy.ndimage.generic_filter(arr, rms_, size=size)


def median(arr, size=5):
    """
    A nonlinear n-D edge-preserving smoothing filter.

    Args:
        arr (ndarray): an n-dimensional array, such as a seismic horizon.
        size (int): the kernel size, e.g. 5 for 5x5. Should be odd,
            rounded up if not.

    Returns:
        ndarray: the resulting smoothed array.
    """
    arr = np.array(arr, dtype=np.float)

    if not size // 2:
        size += 1

    return scipy.ndimage.generic_filter(arr, np.median, size=size)


def mode(arr, size=5, tie='smallest'):
    """
    A nonlinear n-D categorical smoothing filter. Use this to filter non-
    continuous variables, such as categorical integers, e.g. to label facies.

    Args:
        arr (ndarray): an n-dimensional array, such as a seismic horizon.
        size (int): the kernel size, e.g. 5 for 5x5. Should be odd,
            rounded up if not.
        tie (str): `'smallest'` or `'largest`'. In the event of a tie (i.e. two
            or more values having the same count in the kernel), whether to
            give back the smallest of the tying values, or the largest.

    Returns:
        ndarray: the resulting smoothed array.
    """
    def func(this, tie):
        if tie == 'smallest':
            m, _ = scipy.stats.mode(this)
        else:
            m, _ = -scipy.stats.mode(-this)
        return np.squeeze(m)

    arr = np.array(arr, dtype=np.float)

    if not size // 2:
        size += 1

    return scipy.ndimage.generic_filter(arr, func, size=size,
                                        extra_keywords={'tie': tie}
                                       )


def snn(arr, size=5, include=True):
    """
    Symmetric nearest neighbour, a nonlinear 2D smoothing filter.
    http://subsurfwiki.org/wiki/Symmetric_nearest_neighbour_filter

    Args:
        arr (ndarray): a 2D array, such as a seismic horizon.
        size (int): the kernel size, e.g. 5 for 5x5. Should be odd,
            rounded up if not.
        include (bool): whether to include the central pixel itself.

    Returns:
        ndarray: the resulting smoothed array.
    """
    def func(this, pairs, include):
        """
        Deal with this patch.
        """
        centre = this[this.size // 2]
        select = [nearest(this[p], centre) for p in pairs]
        if include:
            select += [centre]
        return np.mean(select)

    arr = np.array(arr, dtype=np.float)
    if arr.ndim != 2:
        raise BrugesError("arr must have 2-dimensions")

    if not size // 2:
        size += 1

    pairs = [[i, size**2-1 - i] for i in range(size**2 // 2)]
    return scipy.ndimage.generic_filter(arr,
                                        func,
                                        size=size,
                                        extra_keywords={'pairs': pairs,
                                                        'include': include}
                                       )


def kuwahara(arr, size=5):
    """
    Kuwahara, a nonlinear 2D smoothing filter.
    http://subsurfwiki.org/wiki/Kuwahara_filter

    Args:
        arr (ndarray): a 2D array, such as a seismic horizon.
        size (int): the kernel size, e.g. 5 for 5x5. Should be odd,
            rounded up if not.

    Returns:
        ndarray: the resulting smoothed array.
    """
    def func(this, s, k):
        """
        Deal with this patch.
        """
        t = this.reshape((s, s))
        sub = np.array([t[:k, :k].flatten(),
                        t[:k, k-1:].flatten(),
                        t[k-1:, :k].flatten(),
                        t[k-1:, k-1:].flatten()]
                      )
        select = sub[np.argmin(np.var(sub, axis=1))]
        return np.mean(select)

    arr = np.array(arr, dtype=np.float)
    if arr.ndim != 2:
        raise BrugesError("arr must have 2-dimensions")

    if not size // 2:
        size += 1

    k = int(np.ceil(size / 2))

    return scipy.ndimage.generic_filter(arr,
                                        func,
                                        size=size,
                                        extra_keywords={'s': size,
                                                        'k': k,
                                                       }
                                       )


def conservative(arr, size=5, supercon=False):
    """
    Conservative, a nonlinear n-D despiking filter. Very conservative! Only
    changes centre value if it is outside the range of all the other values
    in the kernel. Read http://subsurfwiki.org/wiki/Conservative_filter

    Args:
        arr (ndarray): an n-dimensional array, such as a seismic horizon.
        size (int): the kernel size, e.g. 5 for 5x5 (in a 2D arr). Should be
            odd, rounded up if not.
        supercon (bool): whether to be superconservative. If True, replaces
            pixel with min or max of kernel. If False (default), replaces pixel
            with mean of kernel.

    Returns:
        ndarray: the resulting smoothed array.
    """
    def func(this, k, supercon):
        this = this.flatten()
        centre = this[k]
        rest = [this[:k], this[-k:]]
        mi, ma = np.nanmin(rest), np.nanmax(rest)
        if centre < mi:
            return mi if supercon else np.mean(rest)
        elif centre > ma:
            return ma if supercon else np.mean(rest)
        else:
            return centre

    arr = np.array(arr, dtype=np.float)

    if not size // 2:
        size += 1

    k = int(np.floor(size**arr.ndim / 2))

    return scipy.ndimage.generic_filter(arr,
                                        func,
                                        size=size,
                                        extra_keywords={'k': k,
                                                        'supercon': supercon,
                                                       }
                                       )
