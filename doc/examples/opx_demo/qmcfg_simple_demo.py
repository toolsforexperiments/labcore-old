"""
A very basic OPX config
"""

from labcore.opx.config import QMConfig as QMConfig_


class QMConfig(QMConfig_):

    def config_(self):

        params = self.params

        cfg = {
            'version': 1,
        
            # The hardware
            'controllers': {

                # edit this part so the hardware connections match your setup
                'con2': {
                    'type': 'opx1',
                    'analog_outputs': {
                        5: {'offset': 0.0},
                    },
                    'digital_outputs': {
                        1: {},
                    },
                    'analog_inputs': {
                        2: {'offset': 0.0},
                    },
                },
            },
        
            # The logical elements
            'elements': {
        
                'readout': {
                    'singleInput': {
                        'port': ('con2', 5),
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
                    },
                    
                    'outputs': {
                        'out1': ('con2', 2),
                    },
                    
                    'time_of_flight': 188+28,
                    'smearing': 0,
                },
            },
        
            # The pulses
            'pulses': {

                'readout_short_pulse': {
                    'operation': 'measurement',
                    'length': params.readout.short.len(),
                    'waveforms': {
                        'single': 'box_readout_wf'
                    },
                    'digital_marker': 'ON',
                },

            },
        
            # the waveforms
            'waveforms': {
                'const_wf': {
                    'type': 'constant',
                    'sample': 0.1,
                },
                'zero_wf': {
                    'type': 'constant',
                    'sample': 0.0,
                },
                'box_readout_wf': {
                    'type': 'arbitrary',
                    'samples': [0.0] * params.readout.short.buffer() \
                               + [params.readout.short.amp()] * \
                                (params.readout.short.len()-2*params.readout.short.buffer()) \
                               + [0.0] * params.readout.short.buffer(),
                },
            },
        
            'digital_waveforms': {
        
                'ON': {
                    'samples': [(1, 0)]
                },
        
            },
        }
        return cfg
