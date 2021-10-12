"""
Script used to add archive_files feature to the run_and_save_sweep function
"""

import os
from labcore.ddh5 import run_and_save_sweep
from labcore.measurement.sweep import sweep_parameter

WD = os.path.dirname(os.path.realpath(__file__))
datadir = os.path.join(WD, 'data')
sweep = sweep_parameter('x', range(10))

run_and_save_sweep(sweep, datadir, 'test', archive_files=['test_folder', '*.py', '../../../Book1.xlsx'])

