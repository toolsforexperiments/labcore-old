"""general setup file for OPX measurements.

Use by importing and then configuring the options object.
"""

# this is to prevent the OPX logger to also create log messages (results in duplicate messages)
import os
os.environ['QM_DISABLE_STREAMOUTPUT'] = "1"

from typing import Optional, Callable
from dataclasses import dataclass
from functools import partial

from IPython.display import display
import ipywidgets as widgets

# FIXME: only until everyone uses the latest qm packages.
try:
    from qm.QuantumMachinesManager import QuantumMachinesManager, QuantumMachine
except:
    from qm.QuantumMachinesManager import QuantumMachinesManager
    from qm import QuantumMachine

from qm.qua import *

from instrumentserver.helpers import nestedAttributeFromString



from .opx_tools.config import QMConfig
from .opx_tools import sweep as qmsweep
from .opx_tools.mixer import calibrate_mixer, MixerConfig, mixer_of_step, mixer_imb_step

from . import setup_measurements
from .setup_measurements import *

@dataclass
class Options(setup_measurements.Options):
    _qm_config: Optional[QMConfig] = None

    # this is implemented as a property so we automatically set the
    # options correctly everywhere else...
    @property
    def qm_config(self):
        return self._qm_config

    @qm_config.setter
    def qm_config(self, cfg):
        self._qm_config = cfg
        qmsweep.config = cfg

options = Options()
setup_measurements.options = options

@dataclass
class Mixer:
    config: MixerConfig
    qm: Optional[QuantumMachine] = None

    def run_constant_waveform(self):
        with program() as const_pulse:
            with infinite_loop_():
                play('constant', self.config.element_name)
        qmm = QuantumMachinesManager(host=self.config.qmconfig.opx_address,
                                     port=self.config.qmconfig.opx_port)
        qm = qmm.open_qm(self.config.qmconfig(), close_other_machines=False)
        qm.execute(const_pulse)
        self.qm = qm

    def step_of(self, di, dq):
        if self.qm is None:
            raise RuntimeError('No active QuantumMachine.')
        mixer_of_step(self.config, self.qm, di, dq)

    def step_imb(self, dg, dp):
        if self.qm is None:
            raise RuntimeError('No active QuantumMachine.')
        mixer_imb_step(self.config, self.qm, dg, dp)

def add_mixer_config(element_name, analyzer, generator, element_to_param_map=None, **config_kwargs):
    """
    FIXME: add docu (@wpfff)
    TODO: make sure we document the meaning of `element_to_param_map`.
    """
    if element_to_param_map is None:
        element_to_param_map = element_name

    cfg = MixerConfig(
        qmconfig=options.qm_config,
        opx_address=options.qm_config.opx_address,
        opx_port=options.qm_config.opx_port,
        analyzer=analyzer,
        generator=generator,
        if_param=nestedAttributeFromString(options.parameters, f"{element_to_param_map}.IF"),
        offsets_param=nestedAttributeFromString(options.parameters, f"mixers.{element_to_param_map}.offsets"),
        imbalances_param=nestedAttributeFromString(options.parameters, f"mixers.{element_to_param_map}.imbalance"),
        mixer_name=f'{element_name}_IQ_mixer',
        element_name=element_name,
        pulse_name='constant',
        **config_kwargs
    )
    return Mixer(
        config=cfg,
    )


# A simple graphical mixer tuning tool
def mixer_tuning_tool(mixer):
    # widgets for dc offset tuning
    of_step = widgets.FloatText(description='dc of. step:', value=0.01, min=0, max=1, step=0.001)
    iup_btn = widgets.Button(description='I ^')
    idn_btn = widgets.Button(description='I v')
    qup_btn = widgets.Button(description='Q ^')
    qdn_btn = widgets.Button(description='Q v')

    def on_I_up(b):
        mixer.step_of(of_step.value, 0)

    def on_I_dn(b):
        mixer.step_of(-of_step.value, 0)

    def on_Q_up(b):
        mixer.step_of(0, of_step.value)

    def on_Q_dn(b):
        mixer.step_of(0, -of_step.value)

    iup_btn.on_click(on_I_up)
    idn_btn.on_click(on_I_dn)
    qup_btn.on_click(on_Q_up)
    qdn_btn.on_click(on_Q_dn)

    # widgets for imbalance tuning
    imb_step = widgets.FloatText(description='imb. step:', value=0.01, min=0, max=1, step=0.001)
    gup_btn = widgets.Button(description='g ^')
    gdn_btn = widgets.Button(description='g v')
    pup_btn = widgets.Button(description='phi ^')
    pdn_btn = widgets.Button(description='phi v')

    def on_g_up(b):
        mixer.step_imb(imb_step.value, 0)

    def on_g_dn(b):
        mixer.step_imb(-imb_step.value, 0)

    def on_p_up(b):
        mixer.step_imb(0, imb_step.value)

    def on_p_dn(b):
        mixer.step_imb(0, -imb_step.value)

    gup_btn.on_click(on_g_up)
    gdn_btn.on_click(on_g_dn)
    pup_btn.on_click(on_p_up)
    pdn_btn.on_click(on_p_dn)

    # assemble reasonably for display
    ofupbox = widgets.HBox([iup_btn, qup_btn])
    ofdnbox = widgets.HBox([idn_btn, qdn_btn])
    ofbox = widgets.VBox([of_step, ofupbox, ofdnbox])

    imbupbox = widgets.HBox([gup_btn, pup_btn])
    imbdnbox = widgets.HBox([gdn_btn, pdn_btn])
    imbbox = widgets.VBox([imb_step, imbupbox, imbdnbox])

    box = widgets.HBox([ofbox, imbbox])
    display(box)
