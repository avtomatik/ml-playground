import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft


def plot_discrete_fourier_transform(array: np.ndarray) -> None:
    """
    Discrete Fourier Transform

    Parameters
    ----------
    array : np.ndarray
        DESCRIPTION.

    Returns
    -------
    None
        DESCRIPTION.

    """
    # =========================================================================
    # TODO: Refine It
    # =========================================================================
    plt.plot(
        array,
        label="Labor Productivity",
    )
    plt.plot(
        rfft(array),
        "r:",
        label="Fourier Transform",
    )
    plt.grid()
    plt.legend()
    plt.show()
