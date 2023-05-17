import random
import numpy as np

def polyfit2d(x, y, z, kx=3, ky=3, order=None):
    '''
    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Notes
    -----
    Resultant fit can be plotted with:
    np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

    Parameters
    ----------
    x, y: array-like, 1d
        x and y coordinates.
    z: np.ndarray, 2d
        Surface to fit.
    kx, ky: int, default is 3
        Polynomial order in x and y, respectively.
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.

    Returns
    -------
    Return paramters from np.linalg.lstsq.

    soln: np.ndarray
        Array of polynomial coefficients.
    residuals: np.ndarray
    rank: int
    s: np.ndarray

    '''

    # grid coords
    x, y = np.meshgrid(x, y)
    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx+1, ky+1))

    # solve array
    a = np.zeros((coeffs.size, x.size))

    # for each coefficient produce array x^i, y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x**i * y**j
        a[index] = arr.ravel()

    # do leastsq fitting and return leastsq result
    return np.linalg.lstsq(a.T, np.ravel(z), rcond=None)

# z  = []
# n = 16
# for i in range(n):
#     z.append([])
#     for j in range(n):
#         z[-1].append(random.randint(1, 255))
#
# def getMatrix(n, m, z):
#     x = [i for i in range(n)]
#     y = [i for i in range(m)]
#     return polyfit2d(x, y, z, kx=n, ky=m)
#
#
# n, m = 10, 13
# z = []
# for i in range(n):d
#     z.append([])
#     for j in range(m):
#         z[-1].append(random.randint(1, 255))
#
# temp = getMatrix(n, m, z)
# print(temp)


fxy = polyfit2d([i for i in range(10)], [i for i in range(10)], [[random.randint(1, 255) for i in range(10)] for j in range(10)], kx=10, ky=10)
print(fxy)
