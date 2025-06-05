from copy import deepcopy
from typing import Type, Optional, Callable, Tuple, Any, Dict
import numpy as np

from plottr.data.datadict import DataDict, MeshgridDataDict
from plottr.analyzer.fitters.fitter_base import Fit


def batch_fitting(analysis_class, all_data: Dict[Any, Tuple[np.ndarray, np.ndarray]], **kwargs):
    """ fit multiple datafiles of the same analysis class

    Parameters
    ----------
    analysis_class
        the name of the class of the measurement
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
        resorted_dict[param + '_error'] = []

    labels = []
    for label, fit_result in fit_results.items():
        labels.append(label)
        for key, values in fit_result.lmfit_result.params.items():
            resorted_dict[key].append(values.value)
            resorted_dict[key + '_error'].append(values.stderr)
    return labels, resorted_dict


# FIXME: Docstring incomplete
def iterate_over_slices_1d(data: MeshgridDataDict, slice_ax: str):
    slice_idx = data.axes().index(slice_ax)
    d2 = data.reorder_axes(**{slice_ax: len(data.axes()) - 1})
    iter_shape = d2.shape()[:-1]
    for idx in np.ndindex(iter_shape):
        ret = {
            '_axis_names': d2.axes()[:-1],
            '_axis_idxs': idx,
        }
        for d in d2.dependents():
            ret[d] = d2.data_vals(d)[idx]
        for a in d2.axes():
            ret[a] = d2.data_vals(a)[idx]
        yield ret


# FIXME: Docstring incomplete
def batch_fit_1d(data: MeshgridDataDict, fit_class: Type[Fit], dep: str, indep: str, cb: Optional[Callable] = None):
    # TODO: include option to look at the individual fits? include a multipage pdf, for instance...
    # TODO: get back the best fit curves?

    # first: set up a copy of the structure, but omit the axes we fit over.
    indep_ax_idx = data.axes().index(indep)
    axes = deepcopy(data.axes())
    axes.pop(indep_ax_idx)
    result_shape = list(data.shape())  # we also want the shape, up to the axis that'll disappear
    result_shape.pop(indep_ax_idx)
    result = deepcopy(data.structure(include_meta=False))
    del result[indep]
    result[dep]['axes'] = axes
    for d in data.dependents():
        del result[d]

    # next: populate ax values. as first-order approximation, simply use the first values along the
    # dimension we fit
    copy_idx = tuple(slice(None) if i != indep else 0 for i in data.axes())
    for ax in data.axes():
        if ax != indep:
            result[ax]['values'] = data[ax]['values'][copy_idx]

    result['_fit_success'] = {'axes': axes, 'values': np.zeros(result_shape).astype(bool)}
    for data1d in iterate_over_slices_1d(data, indep):
        i = data1d['_axis_idxs']
        x = data1d[indep]
        y = data1d[dep]
        fit = fit_class(x, y)
        fit_result = fit.run()
        result['_fit_success']['values'][i] = fit_result.lmfit_result.success
        for param_name, param in fit_result.lmfit_result.params.items():
            if param_name not in result:
                result[param_name] = {'axes': axes, 'values': np.zeros(result_shape) * np.nan}
                result[param_name + '-stderr'] = {'axes': axes, 'values': np.zeros(result_shape) * np.nan}

            if fit_result.lmfit_result.success:
                result[param_name]['values'][i] = param.value
                result[param_name + '-stderr']['values'][i] = param.stderr

    result.validate()
    return result
