"""Example config for testing the OPX.

Author: Wolfgang Pfaff <wpfaff at illinois dot edu>
"""
import numpy as np
import logging

from labcore.opx.mixer import MixerCalibration
from labcore.opx.config import QMConfig as QMConfig_

logger = logging.getLogger(__name__)


class QMConfig(QMConfig_):

    def config_(self):
        params = self.params  # if we make use of the parameter manager...

        cfg = {
            'version': 1,

            # The hardware
            'controllers': {

                'con2': {
                    'type': 'opx1',
                    'analog_outputs': {
                        1: {'offset': params.mixers.readout.offsets()[0]},  # I
                        2: {'offset': params.mixers.readout.offsets()[1]},  # Q
                        3: {'offset': params.mixers.qubit.offsets()[0]},  # I
                        4: {'offset': params.mixers.qubit.offsets()[1]},  # Q

                    },
                    'digital_outputs': {
                        1: {},
                    },
                    'analog_inputs': {
                        1: {'offset': 0.0},
                        2: {'offset': 0.0}
                    },
                },
            },

            # The logical elements
            'elements': {

                'readout': {
                    'mixInputs': {
                        'I': ('con2', 1),
                        'Q': ('con2', 2),
                        'lo_frequency': params.readout.LO(),
                        'mixer': 'readout_IQ_mixer',
                    },

                    'digitalInputs': {
                        'readout_trigger': {
                            'port': ('con2', 1),
                            'delay': 144,
                            'buffer': 0,
                        },
                    },

                    'intermediate_frequency': params.readout.IF(),

                    'operations': {
                        'readout_short': 'readout_short_pulse',
                        'readout_long': 'readout_long_pulse',
                        'constant': 'constant_pulse',
                    },

                    'outputs': {
                        'out1': ('con2', 1),
                    },

                    'time_of_flight': 188 + 28,
                    'smearing': 0,
                },

                'qubit': {
                    'mixInputs': {
                        'I': ('con2', 3),
                        'Q': ('con2', 4),
                        'lo_frequency': int(params.qubit.LO()),
                        'mixer': 'qubit_IQ_mixer',
                    },
                    'intermediate_frequency': params.qubit.IF(),

                    'operations': {
                        'long_drive': 'long_drive_pulse',
                        'pi_pulse': 'pi_pulse',
                        'constant': 'constant_pulse',
                    },
                },
            },

            # The pulses
            'pulses': {

                'readout_short_pulse': {
                    'operation': 'measurement',
                    'length': params.readout.short.len(),
                    'waveforms': {
                        'I': 'short_readout_wf',
                        'Q': 'zero_wf',
                    },
                    # Integration weights added automatically later.
                    'digital_marker': 'ON',
                },

                'readout_long_pulse': {
                    'operation': 'measurement',
                    'length': params.readout.long.len(),
                    'waveforms': {
                        'I': 'long_readout_wf',
                        'Q': 'zero_wf',
                    },
                    # Integration weights added automatically later.
                    'digital_marker': 'ON',
                },

                'constant_pulse': {
                    'operation': 'control',
                    'length': 1000,
                    'waveforms': {
                        'I': 'const_wf',
                        'Q': 'zero_wf',
                    },
                },

                'long_drive_pulse': {
                    'operation': 'control',
                    'length': params.qubit.drive.long.length(),
                    'waveforms':{
                        'I': 'long_qubit_drive_wf',
                        'Q': 'zero_wf',
                    },
                    'digital_marker': 'ON',
                },

                'pi_pulse':{
                    'operation': 'control',
                    'length': params.qubit.drive.pipulse.sigma() * params.qubit.drive.pipulse.nsigmas(),
                    'waveforms': {
                        'I': 'qubit_pi_pulse_wf',
                        'Q': 'zero_wf',
                    },
                },

            },

            # the waveforms
            'waveforms': {
                'const_wf': {
                    'type': 'constant',
                    'sample': 0.25,
                },
                'long_readout_wf': {
                    'type': 'constant',
                    'sample': params.readout.long.amp(),
                },
                'zero_wf': {
                    'type': 'constant',
                    'sample': 0.0,
                },
                'short_readout_wf': {
                    'type': 'arbitrary',
                    'samples': [0.0] * int(params.readout.short.buffer()) + \
                               [params.readout.short.amp()] * \
                               int(params.readout.short.len() - 2 * params.readout.short.buffer()) + \
                               [0.0] * int(params.readout.short.buffer()),
                },
                'long_qubit_drive_wf':{
                    'type': 'constant',
                    'sample': params.qubit.drive.long.amp(),
                },
                'qubit_pi_pulse_wf': {
                    'type': 'arbitrary',
                    'samples': params.qubit.drive.pipulse.amp() * \
                        (np.exp(-(np.linspace(1,
                                params.qubit.drive.pipulse.sigma() * params.qubit.drive.pipulse.nsigmas(),
                                params.qubit.drive.pipulse.sigma() * params.qubit.drive.pipulse.nsigmas()) - \
                                  params.qubit.drive.pipulse.sigma() * params.qubit.drive.pipulse.nsigmas()//2)**2 / \
                                (2*params.qubit.drive.pipulse.sigma()**2)))
                },
            },

            'digital_waveforms': {

                'ON': {
                    'samples': [(1, 0)]
                },

            },

            'mixers': {
                'readout_IQ_mixer': [
                    {
                        'intermediate_frequency': int(params.readout.IF()),
                        'lo_frequency': int(params.readout.LO()),
                        'correction': MixerCalibration.IQ_imbalance_correction(
                            *params.mixers.readout.imbalance())
                    },
                ],
                'qubit_IQ_mixer': [
                    {
                        'intermediate_frequency': int(params.qubit.IF()),
                        'lo_frequency': int(params.qubit.LO()),
                        'correction': MixerCalibration.IQ_imbalance_correction(
                            *params.mixers.qubit.imbalance())
                    },
                ],
            },
        }
        return cfg
