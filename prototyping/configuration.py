"""Example config for testing the OPX.

Author: Wolfgang Pfaff <wpfaff at illinois dot edu>
"""
import numpy as np


class QMConfig:
    
    def __init__(self):
    
        # define constants here
        # pulse parameters
        self.readout_if = int(50e6)
        self.box_length = 10000
        self.box_buffer = 100
        self.box_amp = 0.4


        
    
    
    def config(self):

        """
        This config file is for an OPX with a cable connecting analog output 1 straight into analog input 1
        """

        ret = {        
            'version': 1,
        
            # The hardware
            'controllers': {
        
                'con1': {
                    'type': 'opx1',
                    'analog_outputs': {
                        1: {'offset': 0.0},
                    },
                    'analog_inputs': {
                        1: {'offset': 0.0},
                    },
                },
            },
        
            # The logical elements
            'elements': {
        
                'readout': {

                    'singleInput': {
                        'port': ('con1', 1),
                    },
                    'intermediate_frequency': self.readout_if,
                    'operations': {
                        'box': 'box_pulse',
                    },
                    
                    'outputs': {
                        'out1': ('con1', 1),
                    },
                    
                    'time_of_flight': 180,
                    'smearing': 0,
                },
            },
        
            # The pulses
            'pulses': {

                'box_pulse': {
                    'operation': 'measurement',
                    'length': self.box_length,
                    'waveforms': {
                        'single': 'box_wf'
                    },
                    'integration_weights': {
                        'box_sin': 'box_sin',
                        'box_cos': 'box_cos',
                    },
                    'digital_marker': 'ON',

                },
            },

            'waveforms': {

                'box_wf': {
                    'type': 'arbitrary',
                    'samples': [0.0] * int(self.box_buffer) + \
                               [self.box_amp] * \
                               int(self.box_length - 2 * self.box_buffer) + \
                               [0.0] * int(self.box_buffer),
                },
            },

            'digital_waveforms': {

                'ON': {
                    'samples': [(1, 0)]
                },
            },
        
            'integration_weights': {

                'box_sin': {
                    'cosine': [0.0] * int(self.box_length),
                    'sine': [1.0] * int(self.box_length),
                },

                'box_cos': {
                    'cosine': [1.0] * int(self.box_length),
                    'sine': [0.0] * int(self.box_length),
                },

            },
        
            'mixers': {
                # 'readout_IQ_mixer': [
                #     {
                #         'intermediate_frequency': self.readout_if,
                #         'lo_frequency': self.readout_lo_frq,
                #         'correction': MixerCalibration.IQ_imbalance_correction(
                #             *self.readout_mixer_imbalance)
                #     },
                # ],
                # 'qubit_IQ_mixer': [
                #     {
                #         'intermediate_frequency': self.qubit_if,
                #         'lo_frequency': self.qubit_lo_frq,
                #         'correction': MixerCalibration.IQ_imbalance_correction(
                #             *self.qubit_mixer_imbalance)
                #     },
                # ],
            },
        }
        
        return ret

