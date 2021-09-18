"""
The following tests are for the run_and_save_sweep function in ddh5.py. I create a simple sweep that creates some dummy
data, then run it and save it using run_and_save_sweep and pass different objects as kwargs to save. I test different
dictionaries to test that they are being saved in the correct data file (JSON if the dictionary can be saved as it) or
pickle anything that is not possible to save as JSON.
"""

import glob
import os
import json
import pickle

import numpy as np
import pytest

from labcore.measurement import *
from labcore.ddh5 import run_and_save_sweep


class NonJsonObject:

    def __init__(self, number):
        self.exists = True
        self.number = number

    def __eq__(self, other):
        if self.number == other.number:
            return True
        else:
            return False


@recording(
    independent('y'),
    dependent('z', depends_on=['y']))
def random_number(start=5, stop=-5, npoints=10, deviation=1):
    y = np.linspace(start, stop, npoints)
    length = len(y)
    z = y ** 2 + deviation * np.random.random(length) * [np.random.choice([-1, 1]) for i in range(length)]
    return y, z


sweep = sweep_parameter('x', range(10), random_number)
simple_dictionary = {
    'state_1': 100,
    'state_2': True,
    'text': 'This is a meaningless string'
}
numpy_array_dictionary = {'numpy array': np.random.random(10)}
nonjson_dictionary = {'non json object': NonJsonObject(number=3)}
mixed_numpy_dictionary = {**simple_dictionary, **numpy_array_dictionary}
nonjson_no_numpy_dictionary = {**simple_dictionary, **nonjson_dictionary}
everything_dictionary = {**simple_dictionary,
                         **numpy_array_dictionary,
                         **nonjson_dictionary}

numpy_array = np.random.random(20)


def test_simple_dictionary_saving(tmpdir):
    datadir = tmpdir
    run_and_save_sweep(sweep, datadir, 'dictionary_saving', metadata=simple_dictionary)
    data_file_path = glob.glob(os.path.join(datadir, '**', '*.ddh5'), recursive=True)

    head, tail = os.path.split(data_file_path[0])
    json_file = os.path.join(head, 'metadata.json')
    if os.path.isfile(json_file):
        with open(json_file, 'r') as f:
            loaded_dictionary = json.load(f)
        assert loaded_dictionary == simple_dictionary
    else:
        assert False

def test_simple_numpy_array_dictionary_saving(tmpdir):
    datadir = tmpdir
    run_and_save_sweep(sweep, datadir, 'saving_numpy_array_in_dictionary', metadata=numpy_array_dictionary)
    data_file_path = glob.glob(os.path.join(datadir, '**', '*.ddh5'), recursive=True)

    head, tail = os.path.split(data_file_path[0])
    json_file = os.path.join(head, 'metadata.json')
    if os.path.isfile(json_file):
        with open(json_file, 'r') as f:
            loaded_dictionary = json.load(f)
        assert loaded_dictionary == numpy_array_dictionary
    else:
        assert False


def test_mixed_dictionary_with_numpy_array(tmpdir):
    datadir = tmpdir
    run_and_save_sweep(sweep, datadir, 'saving_mixed_numpy_array_in_dictionary', metadata=numpy_array_dictionary)
    data_file_path = glob.glob(os.path.join(datadir, '**', '*.ddh5'), recursive=True)

    head, tail = os.path.split(data_file_path[0])
    json_file = os.path.join(head, 'metadata.json')
    if os.path.isfile(json_file):
        with open(json_file, 'r') as f:
            loaded_dictionary = json.load(f)
        assert loaded_dictionary == numpy_array_dictionary
    else:
        assert False


def test_nonjson_dictionary(tmpdir):
    datadir = tmpdir
    run_and_save_sweep(sweep, datadir, 'saving_nonjson_object_in_dictionary', metadata=nonjson_dictionary)
    data_file_path = glob.glob(os.path.join(datadir, '**', '*.ddh5'), recursive=True)
    head, tail = os.path.split(data_file_path[0])
    pickle_file = os.path.join(head, 'metadata.pickle')

    if os.path.isfile(pickle_file):
        with open(pickle_file, 'rb') as f:
            loaded_object = pickle.load(f)
        assert loaded_object['non json object'].number == nonjson_dictionary['non json object'].number
    else:
        assert False


def test_nonjson_mixed_dictionary(tmpdir):
    datadir = tmpdir
    run_and_save_sweep(sweep, datadir, 'saving_numpy_array_in_dictionary', metadata=nonjson_no_numpy_dictionary)
    data_file_path = glob.glob(os.path.join(datadir, '**', '*.ddh5'), recursive=True)
    head, tail = os.path.split(data_file_path[0])
    pickle_file = os.path.join(head, 'metadata.pickle')

    if os.path.isfile(pickle_file):
        with open(pickle_file, 'rb') as f:
            loaded_object = pickle.load(f)
        assert loaded_object == nonjson_no_numpy_dictionary
    else:
        assert False

def test_everything_dictionary(tmpdir):
    datadir = tmpdir
    run_and_save_sweep(sweep, datadir, 'saving_numpy_array_in_dictionary', metadata=everything_dictionary)
    data_file_path = glob.glob(os.path.join(datadir, '**', '*.ddh5'), recursive=True)
    head, tail = os.path.split(data_file_path[0])
    pickle_file = os.path.join(head, 'metadata.pickle')

    if os.path.isfile(pickle_file):
        with open(pickle_file, 'rb') as f:
            loaded_object = pickle.load(f)
        assert loaded_object == everything_dictionary
    else:
        assert False


def test_numpy_array(tmpdir):
    datadir = tmpdir
    run_and_save_sweep(sweep, datadir, 'saving_numpy_array_in_dictionary', metadata=numpy_array)
    data_file_path = glob.glob(os.path.join(datadir, '**', '*.ddh5'), recursive=True)
    head, tail = os.path.split(data_file_path[0])
    pickle_file = os.path.join(head, 'metadata.pickle')

    if os.path.isfile(pickle_file):
        with open(pickle_file, 'rb') as f:
            loaded_object = pickle.load(f)
        assert np.array_equal(loaded_object, numpy_array)
    else:
        assert False
