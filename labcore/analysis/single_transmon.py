from typing import List, Union

import numpy as np
from numpy import complexfloating, ndarray
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from labcore.plotting.basics import readout_hist


def convert_to_probability(signal: List[complexfloating],
                           initial_centroids: List[List[float]] = None,
                           return_labels: bool = False,
                           return_centers: bool = False,
                           plot_hist: bool = False, color_plot: bool = False) -> Union[ndarray, List[ndarray]]:
    """
    Converts IQ data from qubit into state probabilities.

    Analyzes the data from a sweeping, identifies two centers (0 and 1) and assigns each data point to one of them.
    After that, calculate the mean of each repetition and returns the probabilities. The assigment of clusters is done
    through K-means algorithm. The function cannot identify which cluster is the excited state and which cluster is the
     ground state, meaning that the probability might be inverted if initial centroids are not specified.
     The repetition axis must be the outermost axis.

    Parameters
    ----------
    signal:
        Array with the complex data from measurements. All repetitions from the experiment should be here
        and the repetition axis should be the outermost.
    initial_centroids:
        Indicates which center is the ground state and which the excited state. The format is a list containing 2 lists
        composed each of 2 floats, e.g.: ``[[-1,0], [1,0]]``, indicating the IQ coordinates of each center. First center
        corresponds to the ground state and second center corresponds to excited state. If this argument remains
        ``None``, the labels and probabilities might be inverted.
    return_labels:
        If True, a numpy array will be returned as its second item in the same shape as signal
        with 1s and 0s for each element of signal indicating to what cluster that element belongs too.
    return_centers:
        If True, the function will also return as its third item a list with the two centers of the
        clusters (qubit states). Defaults to False.
    plot_hist:
        If True, the function will plot an IQ histogram. defaults to False.
    color_plot:
        If True, the function will plot an IQ histogram with a colored scatter plot on top indicating the
        state of each point. Defaults to False

    Returns
    -------
    Union[ndarray, List[ndarray]]
        If return_labels and return_centers are both False, return an ndarray with the probabilities of the qubit to
        be on a specific state for each point. The function doesn't know which state is excited and ground so the
        probabilities might be inversed.
        If either return_labels or return_centers are True, returns a List with the first item being the
        probabilities. If return_labels is True, the second item will be an ndarray of the shape of signal consisting
        of 0s, or 1s indicating the state of each data point. If return_centers is True, the third item (second if
        return_labels is False) will be 2x2 ndarray with the centers of the 2 states.

    """
    ret = []
    signal_flat = signal.flatten()
    signal_arr = np.stack([signal_flat.real, signal_flat.imag], axis=1)

    if initial_centroids is not None:
        if isinstance(initial_centroids, List):
            initial_centroids = np.array(initial_centroids)

        kmeans = KMeans(n_clusters=2, init=initial_centroids)
    else:
        kmeans = KMeans(n_clusters=2)

    kmeans.fit(signal_arr)

    labels = kmeans.labels_.reshape(signal.shape)
    pvals = labels.mean(axis=0)
    ret.append(pvals)

    if return_labels:
        ret.append(labels)
    if return_centers:
        ret.append(kmeans.cluster_centers_)
    if plot_hist:
        fig = readout_hist(signal_flat, 'Histogram')
    if color_plot:
        fig = readout_hist(signal_flat, 'Sorted Histogram')
        plt.scatter(signal_flat.real, signal_flat.imag, c=kmeans.labels_, cmap='bwr')

    if len(ret) == 1:
        return ret[0]
    else:
        return ret

