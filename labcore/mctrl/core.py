"""
This module contains the most basic elements of labcore.
"""

import collections.abc
from typing import Tuple, List, Any, Iterable, Callable, Dict, Union, Sequence
from functools import wraps
from enum import Enum, auto

from ..utils import same_type


class DataType(Enum):
    """Valid options for data types used in :class:`DataSpec`"""

    #: scalar (single-valued) data. typically numeric, but also bool, etc.
    scalar = auto()
    #: multi-valued data. typically numpy-arrays.
    array = auto()


class SweepStatus(Enum):
    """Options for current sweep status"""

    #: not running yet (instantiated, but iteration hasn't started)
    new = auto()
    #: running
    running = auto()
    #: completed the iteration
    finished = auto()
    #: aborted incomplete
    aborted = auto()


class DataSpec:
    """class to specify data parameters

    For an instance 'obj' of `ParamSpec`, use obj.<property> to access
    any of the properties.

    :param name: name of the parameter
    :param unit: unit of the parameter
    :param datatype: type, see :class:`ParamType`
    :param independent: whether it is an independent parameter (one we control)
        or a dependent one (one we can only read, if you like).
    :param size: the size of the data contained in the parameter. Can be:
        * `None`: entirely unspecified (default)
        * an integer N: a 1-D series of length N. N = -1: unspecified length.
        * a tuple of length k, with entries (N_1, N_2, ...): a k-D series,
          where N_i is the length of dimension i. N_i = -1: length of that
          dimension is not specified.
    """

    def __init__(self, name: str, unit: str = '',
                 datatype: DataType = DataType.scalar,
                 independent: bool = False,
                 size: Union[None, int, Tuple[int]] = None):
        """Constructor for the ParamSpec object"""

        self.name = name
        self.unit = unit
        self.datatype = datatype
        self.independent = independent
        self.size = size

    def __repr__(self):
        return (f"{self.name} ({self.unit}); "
                f"datatype: {self.datatype}; "
                f"independent: {self.independent}; "
                f"size: {self.size}; ")


def dspec(*arg, **kw):
    """Convenience function to create a DataSpec.

    Passes all arguments to the constructor of Data.
    """
    return DataSpec(*arg, **kw)


def return_data(*specs: DataSpec):
    """Returns a decorator that allows adding data parameter specs to a
    function.

    Use with functions that either
    a) return data values: one return value per paramspec given.
       May be one value or a tuple of multiple.
    b) return an iterable: each iteration must give one data value per
       paramspec given.
    """

    def decorator(func):
        func.data_specs = specs

        @wraps(func)
        def wrap(*arg, **kw):
            return func(*arg, **kw)

        return wrap

    return decorator


class Sweep:
    """
    The labcore sweep object.

    After construction, the sweep object can be iterated over.
    Each iteration returns the data returned by pointer and action functions
    during the iteration, as a dictionary in the form
    ```
        {'data parameter name': value, ... }
    ```
    :param action: a function to be executed for each point.
        Returned data parameters must be specified. Use the `@return_data`
        decorator for that.
    :param pointer: a callable that determines the setpoint(s) of the sweep.
        the callable must return an iterable that provides the setpoints.
        it also must provide information about the data parameters returned
        during the iteration. Use the `@return_data` decorator for that.
    :param pointer_opts: a dictionary that will be passed as keyword arguments
        to the pointer function.
    :param action_opts: a dictionary that will be passed as keyword arguments
        to the action function.
    :param state: a dictionary that may contain arbitrary information.
    """

    # TODO: allow multiple action functions
    # TODO: check for conflicting data names. for robustness, warn, but
    #   rename and don't crash
    # TODO: check return compatibility with data specs
    # TODO: is there way to identify the current sweep (if there are multiple).
    #   maybe an ID or so?
    # TODO: system for communicating the total number of points (and index)
    #   within multiple sweeps
    # TODO: need to generate a sweep-level data spec that the measurement can
    #   use

    def __init__(self,
                 action: Union[Callable, List[Callable]],
                 pointer: Callable,
                 pointer_opts: Dict[str, Any] = None,
                 action_opts: Union[Dict[str, Any],
                                    List[Dict[str, Any]]] = None,
                 state: Dict[str, Any] = None):
        """Constructor class for the Sweep object."""

        self.points = None
        self.action = action
        self.pointer = pointer
        if state is None:
            state = dict()
        self.state = state
        if action_opts is None:
            action_opts = dict()
        self.action_opts = action_opts
        if pointer_opts is None:
            pointer_opts = dict()
        self.pointer_opts = pointer_opts

        # this dictionary tracks the state of *this* sweep
        self.sweep_state = dict()
        self.sweep_state['STATUS'] = SweepStatus.new

    def __iter__(self):
        return self

    def __next__(self):

        # this tells us that we're entering the sweep.
        if self.points is None:
            self.sweep_state['CUR_IDX'] = 0
            self.sweep_state['STATUS'] = SweepStatus.running
            self.points = iter(
                self.pointer(sweep_state=self.sweep_state,
                             **self.pointer_opts))
        else:
            self.sweep_state['CUR_IDX'] += 1

        ret_val = {}

        try:
            next_point = next(self.points)
            if not isinstance(next_point, collections.abc.Iterable):
                next_point = [next_point]
            for p, v in zip(self.pointer.data_specs, next_point):
                if p is not None:
                    ret_val[p.name] = v

        except StopIteration:
            self.sweep_state['STATUS'] = SweepStatus.finished
            raise StopIteration

        set_ret = self.action(*next_point, sweep_state=self.sweep_state,
                              **self.action_opts)
        if not isinstance(set_ret, collections.abc.Iterable):
            set_ret = [set_ret]
        for p, v in zip(self.action.data_specs, set_ret):
            ret_val[p.name] = v

        return ret_val


class SweepSpec:
    """
    A class to specify a sweep (without creating it, yet).

    This class is useful because the sweep itself cannot be used further
    after iterating over it once. Having this class allows the
    :class:`Measurement` class to instantiate the actuall sweep objects when
    necessary.

    The parameters are documented in the :class:`Sweep` object.
    """

    def __init__(self, action: Callable, pointer: Callable,
                 pointer_opts: Dict[str, Any] = None,
                 action_opts: Union[Dict[str, Any],
                                    List[Dict[str, Any]]] = None):
        self.pointer = pointer
        self.action = action
        self.pointer_opts = pointer_opts
        self.action_opts = action_opts

    def sweep(self, **kw) -> Sweep:
        """creates and returns the sweep object according to the specs.
        All keyword arguments are passed on to the :class:`Sweep` constructor.
        """
        return Sweep(action=self.action, pointer=self.pointer,
                     pointer_opts=self.pointer_opts,
                     action_opts=self.action_opts, **kw)


def sweepspec(action: Callable,
              pointer: Union[Callable, Iterable,
                             Tuple[Union[DataSpec, Tuple[DataSpec],
                                         Tuple[str]], Tuple[Tuple[str]],
                                   Iterable]],
              **kw) -> SweepSpec:
    """A convenience function to create :class:`SweepSpec` objects.

    :param action: a function to execute at each point.
    :param pointer:
    :returns:
    """
    if isinstance(pointer, Callable):
        ptrfun = pointer

    else:
        # TODO: check if the iterable has a length
        # if an iterable is given, we assume that data values
        # are returned
        if isinstance(pointer, collections.abc.Iterable) \
                and not isinstance(pointer, tuple):
            specs = (dspec('value', independent=True),)
            values = pointer

        # let's check if pointer is a tuple of (specs, iterable) of some sort
        elif isinstance(pointer, tuple):
            if (len(pointer) != 2) or \
                    (not isinstance(pointer[1], collections.abc.Iterable)):
                raise ValueError("Cannot interpret the pointer given.")

            # in any case, the latter entry should be the iterable
            values = pointer[1]

            # Next: inspect the first element to see how the specs were given
            # Note: this part is mainly to translate to DataSpec. Verifying
            #   that a good set of dataspec objects is given should be done
            #   by the SweepSpec or Sweep class.
            #
            # case: a single dataspec object
            if isinstance(pointer[0], DataSpec):
                specs = (pointer[0],)

            # if it's a tuple, it might be multiple things
            elif isinstance(pointer[0], tuple) and len(pointer[0]) > 0:
                tpl = pointer[0]

                # case: a tuple of dataspec objects is clear
                if isinstance(tpl[0], DataSpec) and same_type(*tpl):
                    specs = tpl

                # case: a tuple with a string creates a single dataspec
                # within this function we assume the default for independent
                # to be true -- much more likely for pointer functions
                elif isinstance(tpl[0], str):
                    _kw = dict()
                    if len(tpl) < 4:
                        _kw['independent'] = True
                    specs = (dspec(*tpl, **_kw),)

                # case: a tuple of tuples starting with strings creates
                # multiple specs
                elif isinstance(tpl[0], tuple) and isinstance(tpl[0][0], str):
                    specs = []
                    for t in tpl:
                        _kw = dict()
                        if len(t) < 4:
                            _kw['independent'] = True
                        specs.append(dspec(*t, **_kw))

                else:
                    raise ValueError("Cannot interpret the pointer given.")
            else:
                raise ValueError("Cannot interpret the pointer given.")
        else:
            raise ValueError("Cannot interpret the pointer given.")

        @return_data(*specs)
        def ptrfun(**kw):
            print(specs)
            for p in values:
                yield p

    return SweepSpec(action, ptrfun, **kw)

# class Measurement:
#
#     def __init__(self):
#         self.sweeps = []
#
#     def run(self):
#         pass
