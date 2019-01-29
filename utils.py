import numpy as np
from numpy import linalg as LA
from scipy.misc import factorial


def randomization( P, q, t ):
    result = np.zeros( P.shape )
    # use Fox and Glynn [FG88] to determinae l and r
    l, r = getPoissonBounds(q*t, 1)

    for k in range(l,r):
        e_tmp = np.exp(-q*t)*np.power(q*t, k)/factorial(k)
        e = 0
        if e_tmp < np.inf:
            e = e_tmp
        result = result + e * LA.matrix_power(P, k)

    return result


def getPoissonBounds(alpha, error):
    (L, R, F) = (0, 0, False)
    # cm = (1 / np.sqrt(2 * np.pi * np.floor(alpha))) * np.exp(np.floor(alpha) - alpha - 1 / 12 * np.floor(alpha))
    (low_k, K) = (0, 0)

    if 0 < alpha < 25:
        L = 0
        # F = not (np.exp(-alpha) == -np.inf)

    elif 25 <= alpha:
        b = (1 + (1.0 / alpha)) * np.exp(1.0 / (8 * alpha))

        lower_store = np.inf
        low_k = 3
        while lower_store >= error / 2:
            low_k = low_k + 1
            lower_store = b * np.exp(-(low_k * low_k) / 2) \
                        / (low_k * np.sqrt(2 * np.pi))

        L = np.floor(np.floor(alpha) - (low_k * np.sqrt(alpha)) - 3.0 / 2)

    elif 0 == alpha:
        R = 0
        # F = False

    if alpha > 0:

        l = alpha
        if alpha < 400:
            l = 400

        a = (1 + (1.0 / l)) * np.exp(1.0 / 16) * np.sqrt(2)

        upper_store = np.inf
        k = 3
        while upper_store >= error / 2:
            k = k + 1
            d = 1.0 / (1 - np.exp(-(2.0 / 9) * (k * np.sqrt(2 * alpha) + 3.0 / 2)))
            upper_store = a * d * np.exp(-(k * k) / 2) \
                         / (k * np.sqrt(2 * np.pi))

        R = (np.ceil(np.floor(alpha) + (k * np.sqrt(2 * l)) + 3.0 / 2.0))

    return int(L*1.5), int(R*.5)

