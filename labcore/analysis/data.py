"""Tools for more convenient data handling."""


from typing import Union, List, Optional, Type
from types import TracebackType
from pathlib import Path
from datetime import datetime
import json

from matplotlib.figure import Figure
from matplotlib import pyplot as plt

from plottr.data.datadict import DataDictBase, datadict_to_meshgrid, MeshgridDataDict
from plottr.data.datadict_storage import datadict_from_hdf5


def data_info(folder: str, fn: str = 'data.ddh5', do_print: bool = True):
    fn = Path(folder, fn)
    dataset = datadict_from_hdf5(fn)
    if do_print:
        print(dataset)
    else:
        return str(dataset)


def get_data(folder: Union[str, Path], data_name: Optional[Union[str, List[str]]] = None, fn: str = 'data.ddh5',
             mk_grid: bool = True, avg_over: Optional[str] = 'repetition') -> DataDictBase:

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
    dataset = dataset.extract(data_name)

    if mk_grid:
        dataset = datadict_to_meshgrid(dataset)

    if avg_over is not None and avg_over in dataset.axes() and isinstance(dataset, MeshgridDataDict):
        if not dataset.axes_are_compatible():
            raise RuntimeError('When averaging, axes must be compatible.')

        avg_ax_id = dataset.axes(data_name[0]).index(avg_over)
        for dn, _ in dataset.data_items():
            dataset[dn]['values'] = dataset[dn]['values'].mean(axis=avg_ax_id)
            if avg_over in dataset[dn]['axes']:
                dataset[dn]['axes'].pop(avg_ax_id)
        del dataset[avg_over]

    dataset.validate()
    return dataset


class DatasetAnalysis:

    def __init__(self, folder):
        self.figure_save_format = ['png', 'pdf']
        self.folder = folder
        if not isinstance(self.folder, Path):
            self.folder = Path(self.folder)
        self.timestamp = str(datetime.now().replace(microsecond=0).isoformat().replace(':', ''))

        self.figures = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        pass

    def save(self):
        for n, f in self.figures.items():
            self.save_figure(f, n)

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

    def make_figure(self, name, *arg, **kwargs):
        if name in self.figures:
            raise ValueError('figure with that name already exists in this analysis')
        fig = plt.figure(*arg, **kwargs)
        self.figures[name] = fig
        return fig

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
            fn = Path(self.folder, f"{self.timestamp}_{name}.{f}")
            fig.savefig(fn)

