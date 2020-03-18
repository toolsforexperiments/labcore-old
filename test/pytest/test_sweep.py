"""Tests for mctrl sweeps"""
import numpy as np
from labcore.mctrl.core import Sweep, dspec, return_data, SweepStatus, sweepspec


def test_basic_sweep():
    """run a full sweep with various numbers of data params.
    Test essential forwarding of information and data.
    """
    # TODO: may be more robust with random matrix data.

    for npts in [0, 10, 1000]:
        for nptparams in [0, np.random.randint(1, 10)]:
            for nactionparams in [0, np.random.randint(1, 10)]:

                ptparams = [dspec(f'point_param-{i}') for i in range(nptparams)]
                actionparams = [dspec(f'action_param-{i}') for i in
                                range(nactionparams)]

                @return_data(*ptparams)
                def pointfun(*arg, **kw):
                    for i in range(npts):
                        yield tuple(i + j for j in range(nptparams))

                @return_data(*actionparams)
                def actionfun(*arg, **kw):
                    idx = kw['sweep_state']['CUR_IDX']

                    # test that the arguments are passed correctly
                    assert len(arg) == nptparams
                    assert list(arg) == list(idx + j for j in range(nptparams))

                    return tuple(j for j in range(5, 5 + nactionparams))

                sweep = Sweep(pointer=pointfun, action=actionfun)
                assert sweep.sweep_state['STATUS'] == SweepStatus.new

                # iterate through sweep and check that returned data is correct
                for k, ret in enumerate(sweep):
                    assert sweep.sweep_state['CUR_IDX'] == k
                    if k < npts - 1:
                        assert sweep.sweep_state[
                                   'STATUS'] == SweepStatus.running
                    for j in range(nptparams):
                        assert ret[f'point_param-{j}'] == k + j
                    for j in range(nactionparams):
                        assert ret[f'action_param-{j}'] == j + 5

                assert sweep.sweep_state['STATUS'] == SweepStatus.finished


def test_sweep_spec():
    """create a sweep directly and using mksweep; compare if identical."""

    @return_data(dspec('a'))
    def avals(num=3, **kw):
        for i in range(num):
            yield i

    @return_data()
    def set_a(value, **kw):
        assert value == kw['sweep_state']['CUR_IDX']

    sweep_spec = sweepspec(set_a, avals, pointer_opts=dict(num=5))
    sweep_from_spec = sweep_spec.sweep()

    sweep_from_class = Sweep(set_a, avals, pointer_opts=dict(num=5))

    for a, b in zip(sweep_from_class, sweep_from_spec):
        assert a == b
