"""plottr.data.datadict_storage

Provides file-storage tools for the DataDict class.

Description of the HDF5 storage format
======================================

We use a simple mapping from DataDict to the HDF5 file. Within the file,
a single DataDict is stored in a (top-level) group of the file.
The data fields are datasets within that group.

Global meta data of the DataDict are attributes of the group; field meta data
are attributes of the dataset (incl., the `unit` and `axes` values). The meta
data keys are given exactly like in the DataDict, i.e., incl the double
underscore pre- and suffix.
"""
import os
import time
from enum import Enum
from typing import Any, Union, Optional, Dict, Type, Collection
from types import TracebackType

import numpy as np
import h5py
import json
import pickle

from plottr.data.datadict import DataDict, is_meta_key
from plottr.data.datadict_storage import *

from .measurement.sweep import Sweep

__author__ = 'Wolfgang Pfaff'
__license__ = 'MIT'


def _create_datadict_structure(sweep: Sweep) -> DataDict:
    """
    Returns a structured DataDict from the DataSpecs of a Sweep.

    :param sweep: Sweep object from which the DataDict is created.
    """

    data_specs = sweep.get_data_specs()
    data_dict = DataDict()
    for spec in data_specs:

        depends_on = spec.depends_on
        unit = spec.unit
        name = spec.name

        # Checks which fields have information and which ones are None.
        if depends_on is None:
            if unit is None:
                data_dict[name] = dict()
            else:
                data_dict[name] = dict(unit=unit)
        else:
            if unit is None:
                data_dict[name] = dict(axes=depends_on)
            else:
                data_dict[name] = dict(axes=depends_on, unit=unit)

    data_dict.validate()

    return data_dict


def _check_none(line: Dict) -> bool:
    """
    Checks if the values in a Dict are all None. Returns True if all values are None, False otherwise.
    """
    for arg in line.keys():
        if line[arg] is not None:
            return False
    return True

def _save_dictionary(dict: Dict, filepath: str) -> None:

    with open(filepath, 'w') as f:
        json.dump(dict, f, indent=2, sort_keys=True)

def _pickle_and_save(obj, filepath: str) -> None:

    try:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    except TypeError as pickle_error:
        print(f'Object could not be pickled: {pickle_error.args}')


def run_and_save_sweep(sweep: Sweep, data_dir: str, name: str, **extra_saving_items) -> None:
    """
    Iterates through a sweep, saving the data coming through it into a file called <name> at <data_dir> directory.

    :param sweep: Sweep object to iterate through.
    :param data_dir: Directory of file location.
    :param name: Name of the file.
    :param extra_saving_items: Kwargs for extra objects that should be saved. If the kwarg is a dictionary, the function
        will try and save it as a JSON file. If the dictionary contains objects that are not JSON serializable it will
        be pickled. Any other kind of object will be pickled too. The files will have their keys as names.

    """
    data_dict = _create_datadict_structure(sweep)

    # Creates a file even when it fails.
    with DDH5Writer(data_dict, data_dir, name=name) as writer:

        # Saving meta-data
        dir = writer.filepath.removesuffix(writer.filename)
        for key, value in extra_saving_items.items():
            dictionary_dir = dir + '\\' + key + '.json'
            if isinstance(value, dict):
                try:
                    _save_dictionary(value, dictionary_dir)
                except TypeError as error:
                    # Delete the file created by _save_dictionary. This file does not contain the complete dictionary.
                    if os.path.isfile(dictionary_dir):
                        os.remove(dictionary_dir)

                    converted = False # Flag to see if there has been a converted ndarray.
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            value[k] = v.tolist()
                            converted = True
                    if converted:
                        try:
                            _save_dictionary(value, dir + '\\' + key + '.json')
                        except TypeError as e:

                            if os.path.isfile(dictionary_dir):
                                os.remove(dictionary_dir)

                            print(f'{key} has not been able to save to json: {e.args}.'
                                  f' The item will be pickled instead.')
                            _pickle_and_save(value, dir + '\\' + key + '.pickle')
                    else:
                        print(f'{key} has not been able to save to json: {error.args}.'
                              f' The item will be pickled instead.')
                        _pickle_and_save(value, dir + '\\' + key + '.pickle')
            else:
                _pickle_and_save(value, dir + '\\' + key + '.pickle')


        # Save data.
        for line in sweep:
            if not _check_none(line):
                writer.add_data(**line)

    print('The measurement has finished successfully and all of the data has been saved.')
