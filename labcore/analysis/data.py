"""Tools for more convenient data handling."""

import os
import re
from typing import Union, List, Optional, Type, Any, Dict
from types import TracebackType
from pathlib import Path
from datetime import datetime
import json
import logging

import numpy as np
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from plottr.analyzer.base import AnalysisResult
from plottr.analyzer.fitters.fitter_base import FitResult
from plottr.data.datadict import DataDictBase, datadict_to_meshgrid, MeshgridDataDict
from plottr.data.datadict_storage import datadict_from_hdf5


logger = logging.getLogger(__name__)


def data_info(folder: str, fn: str = 'data.ddh5', do_print: bool = True):
    fn = Path(folder, fn)
    dataset = datadict_from_hdf5(fn)
    if do_print:
        print(dataset)
    else:
        return str(dataset)


def timestamp_from_path(p: Path) -> datetime:
    """Return a `datetime` timestamp from a standard-formatted path.
    Assumes that the path stem has a timestamp that begins in ISO-like format
    ``YYYY-mm-ddTHHMMSS``.
    """
    timestring = str(p.stem)[:13] + ":" + str(p.stem)[13:15] + ":" + str(p.stem)[15:17]
    return datetime.fromisoformat(timestring)


def find_data(root,
              newer_than: Optional[datetime]=None,
              older_than: Optional[datetime]=None,
              folder_filter: Optional[str]=None) -> List[Path]:

    folders = []
    for f, dirs, files in os.walk(root):
        if 'data.ddh5' in files:
            fp = Path(f)
            ts = timestamp_from_path(fp)
            if newer_than is not None and ts <= newer_than:
                continue
            if newer_than is not None and ts >= older_than:
                continue
            if folder_filter is not None:
                pattern = re.compile(folder_filter)
                if not pattern.match(str(fp.stem)):
                    continue

            folders.append(fp)
    return sorted(folders)


def get_data(
        folder: Union[str, Path],
        data_name: Optional[Union[str, List[str]]] = None,
        fn: str = 'data.ddh5',
        mk_grid: bool = True,
        avg_over: Optional[str] = 'repetition',
        rotate_complex: bool = False,
    ) -> DataDictBase:

    """Get data from disk.

    Parameters
    ----------
    folder
        the folder containing the data file (a ddh5 file)
    data_name
        which dependent(s) to extract from the data.
        if ``None``, return all data.
    fn
        the file name
    mk_grid
        if True, try to automatically place data on grid.
    avg_over
        if not ``None``, average over this axis if it exists.
    rotate_complex
        if True: try to rotate data automatically in IQ to map onto a single
        axis and return real data. We use sklearn's PCA tool for that.
        this is done after averaging.

    Returns
    -------
    the resulting dataset

    """
    fn = Path(folder, fn)
    dataset = datadict_from_hdf5(fn)
    dataset.add_meta('dataset.folder', str(Path(folder)))
    dataset.add_meta('dataset.filepath', str(fn))

    if data_name is None:
        data_name = dataset.dependents()
    elif isinstance(data_name, str):
        data_name = [data_name]
    dataset = dataset.extract(data_name, copy=False)

    if mk_grid:
        dataset = datadict_to_meshgrid(dataset, copy=False)

    if avg_over is not None and avg_over in dataset.axes() and isinstance(dataset, MeshgridDataDict):
        dataset = dataset.mean(avg_over)

    if rotate_complex:
        for d in data_name:
            dvals = dataset.data_vals(d)
            shp = dvals.shape
            if np.iscomplexobj(dvals):
                pca = PCA(n_components=1)
                newdata = pca.fit_transform(
                    np.vstack((dvals.real.flatten(), dvals.imag.flatten())).T
                    )
                dataset[d]['values'] = newdata.reshape(shp)

    dataset.validate()
    return dataset


class DatasetAnalysis:

    def __init__(self, folder):
        self.figure_save_format = ['png', 'pdf']
        self.folder = folder
        if not isinstance(self.folder, Path):
            self.folder = Path(self.folder)
        self.timestamp = str(datetime.now().replace(microsecond=0).isoformat().replace(':', ''))

        self.additionals = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        pass

    def _new_file_path(self, name: str, suffix: str = '') -> Path:
        if suffix != '':
            name = name + '.' + suffix
        return Path(self.folder, f"{self.timestamp}_{name}")

    # --- loading measurement data --- #
    def get_data(self, data_name, *arg, **kw):
        return get_data(self.folder, data_name, *arg, **kw)

    def load_saved_parameter(self, parameter_name,
                             parameter_manager_name='parameter_manager',
                             file_name='parameters.json'):
        fn = Path(self.folder) / file_name
        with open(fn, 'r') as f:
            data = json.load(f)

        parameter_path = f"{parameter_manager_name}.{parameter_name}"
        if parameter_path not in data:
            raise ValueError('this parameter was not found in the saved meta data.')

        return data[parameter_path]['value']

    # --- Adding analysis results --- #
    def add(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self.additionals:
                raise ValueError('element with that name already exists in this analysis.')
            self.additionals[k] = v

    def add_figure(self, name, *arg, fig: Optional[None], **kwargs) -> Figure:
        if name in self.figures:
            raise ValueError('element with that name already exists in this analysis.')
        if fig is None:
            fig = plt.figure(*arg, **kwargs)
        self.additionals[name] = fig
        return fig

    make_figure = add_figure

    # --- Saving analysis results --- #
    def save(self):
        for name, element in self.additionals.items():
            if isinstance(element, Figure):
                fp = self.save_figure(element, name)

            elif isinstance(element, AnalysisResult):
                fp = self.save_add_dict_data(element.params_to_dict(), name+"_params")
                if isinstance(element, FitResult):
                    fp = self.save_add_str(element.lmfit_result.fit_report(), name+"_lmfit_report")

            elif isinstance(element, np.ndarray):
                fp = self.save_add_np(element, name)

            elif isinstance(element, dict):
                fp = self.save_add_dict_data(element, name)

            elif isinstance(element, str):
                fp = self.save_add_str(element, name)

            else:
                logger.error(f"additional data '{name}' is not supported for saving!")

    def save_figure(self, fig: Figure, name: str):
        """save a figure in a standard way to the dataset directory.

        Parameters
        ----------
        fig
            the figure instance
        name
            name to give the figure
        fmt
            file format (defaults to png)

        Returns
        -------
        ``None``

        """
        fmts = self.figure_save_format
        if not isinstance(fmts, list):
            fmts = [fmts]

        fig.suptitle(f"{self.folder.name}: {name}", fontsize='small')

        for f in fmts:
            fp = self._new_file_path(name, f)
            fig.savefig(fp)

        return fp

    def save_add_dict_data(self, data: dict, name: str):
        fp = self._new_file_path(name, 'json')
        # d = dict_arrays_to_list(data)
        with open(fp, 'x') as f:
            json.dump(data, f, cls=NumpyEncoder)
        return fp

    def save_add_str(self, data: str, name: str):
        fp = self._new_file_path(name, 'txt')
        with open(fp, 'x') as f:
            f.write(data)
        return fp

    def save_add_np(self, data: np.ndarray, name: str):
        fp = self._new_file_path(name, 'json')
        with open(fp, 'x') as f:
            json.dump({name: data}, f, cls=NumpyEncoder)

    # --- loading (and managing) earlier analysis results --- #
    # TBD...



# enable saving numpy to json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
