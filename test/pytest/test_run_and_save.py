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
import shutil

import numpy as np

from labcore.sweep import recording, independent, dependent, sweep_parameter
from labcore.sweep.ddh5 import run_and_save_sweep


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

# Saving meta-data tests.
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
        np.testing.assert_equal(loaded_dictionary, numpy_array_dictionary)
    else:
        assert False


def test_mixed_dictionary_with_numpy_array(tmpdir):
    datadir = tmpdir
    run_and_save_sweep(sweep, datadir, 'saving_mixed_numpy_array_in_dictionary', metadata=mixed_numpy_dictionary)
    data_file_path = glob.glob(os.path.join(datadir, '**', '*.ddh5'), recursive=True)

    head, tail = os.path.split(data_file_path[0])
    json_file = os.path.join(head, 'metadata.json')
    if os.path.isfile(json_file):
        with open(json_file, 'r') as f:
            loaded_dictionary = json.load(f)
        np.testing.assert_equal(loaded_dictionary, mixed_numpy_dictionary)
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
        np.testing.assert_equal(loaded_object, nonjson_dictionary)
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
        np.testing.assert_equal(loaded_object, everything_dictionary)
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

# Archive files tests
def test_archive_specific_file(tmpdir):
    WD = os.getcwd()
    txt_file = os.path.join(WD, 'test_file.txt')
    with open(txt_file, 'w') as f:
        f.write('this is a test file')

    run_and_save_sweep(sweep, tmpdir, 'txt_archive_file', archive_files='test_file.txt')

    os.remove(txt_file)

    data_file_path = glob.glob(os.path.join(tmpdir, '**', '*.ddh5'), recursive=True)
    head, tail = os.path.split(data_file_path[0])
    archived_test_file = os.path.join(head, 'archive_files', 'test_file.txt')
    assert os.path.isfile(archived_test_file)

def test_archive_multiple_files(tmpdir):
    WD = os.getcwd()
    files = [os.path.join(WD, f'test_file#{i}.txt') for i in range(5)]
    for count, file in enumerate(files):
        with open(file, 'w') as f:
            f.write('this is a test file')

    run_and_save_sweep(sweep, tmpdir, 'multiple_files', archive_files=['*.txt'])

    for file in files:
        os.remove(file)

    data_file_path = glob.glob(os.path.join(tmpdir, '**', '*.ddh5'), recursive=True)
    head, tail = os.path.split(data_file_path[0])
    archived_files = [os.path.join(head, 'archive_files', f'test_file#{i}.txt') for i in range(5)]
    num_existing_files = 0
    for ar_file in archived_files:
        if os.path.isfile(ar_file):
            num_existing_files += 1
    assert num_existing_files == 5

def test_archive_folder(tmpdir):
    WD = os.getcwd()
    os.mkdir('test_folder')
    txt_files = [os.path.join(WD, 'test_folder', f'test_file#{i}.txt') for i in range(5)]
    script_files = [os.path.join(WD,'test_folder', f'test_script#{i}.py') for i in range(5)]
    files = txt_files + script_files
    for count, file in enumerate(files):
        with open(file, 'w') as f:
            f.write('this is a test file')

    run_and_save_sweep(sweep, tmpdir, 'folder', archive_files=['test_folder'])

    shutil.rmtree('test_folder')

    data_file_path = glob.glob(os.path.join(tmpdir, '**', '*.ddh5'), recursive=True)
    head, tail = os.path.split(data_file_path[0])
    archived_txt_files = [os.path.join(head, 'archive_files', 'test_folder', f'test_file#{i}.txt') for i in range(5)]
    archived_scripts = [os.path.join(head, 'archive_files', 'test_folder', f'test_script#{i}.py') for i in range(5)]
    archived_files = archived_txt_files + archived_scripts
    num_existing_files = 0
    for ar_file in archived_files:
        if os.path.isfile(ar_file):
            num_existing_files += 1
    assert num_existing_files == 10

def test_archive_everythin(tmpdir):
    WD = os.getcwd()
    os.mkdir('test_folder')
    txt_folder_files = [os.path.join(WD, 'test_folder', f'test_file#{i}.txt') for i in range(5)]
    script_folder_files = [os.path.join(WD, 'test_folder', f'test_script#{i}.py') for i in range(5)]
    txt_files = [os.path.join(WD, f'test_file#{i}.txt') for i in range(5)]
    csv_file = [os.path.join(WD, f'../test_csv.csv')]
    files = txt_folder_files + script_folder_files + txt_files + csv_file
    for file in files:
        with open(file, 'w') as f:
            f.write('this is a test file')

    run_and_save_sweep(sweep, tmpdir, 'folder', archive_files=['*.txt','../test_csv.csv','test_folder'])

    shutil.rmtree('test_folder')
    for file in txt_files + csv_file:
        os.remove(file)

    data_file_path = glob.glob(os.path.join(tmpdir, '**', '*.ddh5'), recursive=True)
    head, tail = os.path.split(data_file_path[0])
    archived_folder_txt_files = [os.path.join(head, 'archive_files', 'test_folder', f'test_file#{i}.txt') for i in range(5)]
    archived_folder_scripts = [os.path.join(head, 'archive_files', 'test_folder', f'test_script#{i}.py') for i in range(5)]
    archived_txt_files = [os.path.join(head,'archive_files', f'test_file#{i}.txt') for i in range(5)]
    archived_csv_file = [os.path.join(head, 'archive_files', f'test_csv.csv')]
    archived_files = archived_folder_txt_files + archived_folder_scripts + archived_csv_file + archived_txt_files
    num_existing_files = 0
    for ar_file in archived_files:
        if os.path.isfile(ar_file):
            num_existing_files += 1
    assert num_existing_files == 16
