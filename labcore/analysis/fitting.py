from typing import Dict, Any, Tuple
import numpy as np


def batch_fitting(analysis_class, all_data: Dict[Any, Tuple[np.ndarray, np.ndarray]], **kwargs):
    """ fit multiple datafiles of the same analysis class

    Parameters
    ----------
    analysis_class
        the name of the class of the sweeping
    all_data
        a dictionary whose values are tuples of coordinates and measured data, each of which corresponds to a data
        file, and keys are the labels for the data files

    Returns
    -------
    dict
        a dictionary whose values are FitResult class objects containing the fit results of the data files,
        and keys are the labels for the data files
    """
    fit_results = {}
    for label in all_data.keys():
        coordinates, data = all_data[label]
        fit = analysis_class(coordinates, data)
        fit_result = fit.run(**kwargs)
        fit_results[label] = fit_result

    return fit_results


def params_from_batch_fitting_results(fit_results: dict, params: list[str]) -> Tuple[list[Any], Dict[str, Any]]:
    """ extract parameter values and errors from the fit results and resort them into a dictionary whose keys are
    the parameter and values are corresponding parameter values and errors

    Parameters
    ----------
    fit_results
        a dictionary whose values are FitResult class objects containing the fit results of the data files,
        and keys are the labels for the data files
    params
        list of strings corresponding to the names of parameters one wants to look at

    Returns
    -------
    dict
        a dictionary whose keys are the parameter and values are corresponding parameter values and errors
    """
    resorted_dict = {}
    for param in params:
        resorted_dict[param] = []
        resorted_dict[param +'_error'] = []

    labels = []
    for label, fit_result in fit_results.items():
        labels.append(label)
        for key, values in fit_result.lmfit_result.params.items():
            resorted_dict[key].append(values.value)
            resorted_dict[key +'_error'].append(values.stderr)
    return labels, resorted_dict
