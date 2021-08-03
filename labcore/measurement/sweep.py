import itertools
from typing import Iterable, Callable, Union, Tuple, Any, Optional, Dict, List
import collections
import logging

try:
    from qcodes import Parameter as QCParameter
    QCODES_PRESENT = True
except ImportError:
    QCParameter = None
    QCODES_PRESENT = False

from .record import produces_record, DataSpec, IteratorToRecords, \
    DataSpecFromTupleType, record_as, combine_data_specs, independent, \
    make_data_spec, data_specs_label
from ..utils import indent_text


logger = logging.getLogger(__name__)


if QCODES_PRESENT:
    ParamSpecType = Union[str, QCParameter, DataSpecFromTupleType, DataSpec]
else:
    ParamSpecType = Union[str, DataSpecFromTupleType, DataSpec]


def once(action: Callable) -> "Sweep":
    """Return a sweep that executes the action once."""
    return Sweep(null_pointer, action)


def sweep_parameter(param: ParamSpecType, sweep_iterable: Iterable,
                    *actions: Callable) -> "Sweep":
    """Create a sweep over a parameter.

    :param param: one of:
        - a string: generates an independent, scalar data parameter
        - a tuple or list: will be passed to the constructor of
          :class:`.DataSpec`; see :func:`.make_data_spec`.
        - a :class:`.DataSpec` instance.
        - a qcodes parameter. In this case the parameter's ``set`` method is
          called for each value during the iteration.
    :param sweep_iterable: an iterable that generates the values the parameter
        will be set to.
    :param actions: an arbitrary number of action functions.
    """

    if isinstance(param, str):
        param_ds = independent(param)
    elif isinstance(param, (tuple, list)):
        param_ds = make_data_spec(*param)
    elif isinstance(param, DataSpec):
        param_ds = param
    elif QCODES_PRESENT and isinstance(param, QCParameter):
        param_ds = independent(param.name, unit=param.unit)

        def setfunc(*args, **kwargs):
            param.set(kwargs.get(param.name))

        actions = list(actions)
        actions.insert(0, setfunc)
    else:
        raise TypeError(f"Cannot make parameter from type {type(param)}")

    record_iterator = IteratorToRecords(sweep_iterable, param_ds)
    return Sweep(record_iterator, *actions)


null_pointer = [None]


def null_action():
    return None


class Sweep:
    """Base class for sweeps.

    Can be iterated over; for each pointer value the associated actions are
    executed. Each iteration step produces a record, containing all values
    produced by pointer and actions that have been annotated as such.
    (see: :func:`.record_as`)

    :param pointer: An iterable that defines the steps over which we iterate
    :param actions: a variable number of functions. Each will be called for
        each iteration step, with the pointer value(s) as arguments, provided
        the function can accept it.
    """

    # TODO: (MAYBE?) should we check if there are conflicting record parameters?
    # TODO: some way of passing meta-info around (about the sweep state)
    #   probably nice to have some info on benchmarking, current indices, etc.
    # TODO: need a way to look through all subsweeps, for example to find all
    #   kw arguments of all actions recursively.
    # TODO: support qcodes parameters as action directly.

    # TODO: these flags should maybe be passed on to 'child' sweeps?
    record_none = True
    pass_on_returns = True
    pass_on_none = False

    @staticmethod
    def link_sweep_properties(src: "Sweep", target: "Sweep"):
        """Share state properties between sweeps."""

        for p in ['_state', '_pass_kwargs', '_action_kwargs']:
            if hasattr(src, p):
                setattr(target, p, getattr(src, p))

    def __init__(self, pointer: Optional[Iterable], *actions: Callable):
        """Constructor of :class:`.Sweep`."""
        self._state = {}
        self._pass_kwargs = {}
        self._action_kwargs = {}

        if pointer is None:
            self.pointer = null_pointer
        elif isinstance(pointer, (collections.abc.Iterable, Sweep)):
            self.pointer = pointer
        else:
            raise TypeError('pointer needs to be iterable.')

        self.actions = []
        for a in actions:
            self.append_action(a)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value: Dict[str, Any]):
        for k, v in value.items():
            self._state[k] = v

    @property
    def pass_kwargs(self):
        return self._pass_kwargs

    @pass_kwargs.setter
    def pass_kwargs(self, value: Dict[str, Any]):
        for k, v in value.items():
            self._pass_kwargs[k] = v

    @property
    def action_kwargs(self):
        return self._action_kwargs

    @action_kwargs.setter
    def action_kwargs(self, value: Dict[str, Any]):
        for k, v in value.items():
            self._action_kwargs[k] = v

    def __iter__(self):
        return self.run()

    def __add__(self, other: Union[Callable, "Sweep"]) -> "Sweep":
        if isinstance(other, Sweep):
            sweep2 = other
        elif callable(other):
            sweep2 = Sweep(None, other)
        else:
            raise TypeError(f'can only combine with Sweep or callable, '
                            f'not {type(other)}')

        Sweep.link_sweep_properties(self, sweep2)
        return append_sweeps(self, sweep2)

    def __mul__(self, other: Union[Callable, "Sweep"]) -> "Sweep":
        if isinstance(other, Sweep):
            sweep2 = other
        elif callable(other):
            sweep2 = Sweep(self.pointer, other)
        else:
            raise TypeError(f'can only combine with Sweep or callable, '
                            f'not {type(other)}')

        Sweep.link_sweep_properties(self, sweep2)
        return zip_sweeps(self, sweep2)

    def __matmul__(self, other: Union[Callable, "Sweep"]) -> "Sweep":
        if isinstance(other, Sweep):
            sweep2 = other
        elif callable(other):
            sweep2 = Sweep(None, other)
        else:
            raise TypeError(f'can only combine with Sweep or callable, '
                            f'not {type(other)}')

        Sweep.link_sweep_properties(self, sweep2)
        return nest_sweeps(self, sweep2)

    def append_action(self, action: Callable):
        """Add an action to the sweep."""
        if callable(action):
            if produces_record(action):
                self.actions.append(action)
            else:
                self.actions.append(record_as(action))
        else:
            raise TypeError('action must be a callable.')

    def run(self) -> "SweepIterator":
        """Create the iterator for the sweep."""
        return SweepIterator(
            self,
            state=self.state,
            pass_kwargs=self.pass_kwargs,
            action_kwargs=self.action_kwargs)

    def set_action_opts(self, **action_kwargs: Dict[str, Any]):
        """configure the sweep actions

        :param action_kwargs: Keyword arguments to pass to action functions
            format: {'<action_name>': {'key': 'value'}
            <action_name> is what action_function.__name__ returns.
        """
        for func, val in action_kwargs.items():
            self.action_kwargs[func] = val

    def get_data_specs(self) -> Tuple[DataSpec, ...]:
        """Return the data specs of the sweep."""
        specs = []
        pointer_specs = []
        if produces_record(self.pointer):
            pointer_specs = self.pointer.get_data_specs()
            specs = combine_data_specs(*(list(specs) + list(pointer_specs)))

        for a in self.actions:
            if produces_record(a):
                action_specs = a.get_data_specs()
                pointer_independents = [ds.name for ds in pointer_specs
                                        if ds.depends_on is None]
                for aspec in action_specs:
                    aspec_ = aspec.copy()
                    if aspec_.depends_on is not None:
                        aspec_.depends_on = pointer_independents + aspec_.depends_on

                    specs = combine_data_specs(*(list(specs) + [aspec_]))

        return tuple(specs)

    def __repr__(self):
        ret = self.pointer.__repr__()
        for a in self.actions:
            ret += f" >> {a.__repr__()}"
        ret += f"\n==> {data_specs_label(*self.get_data_specs())}"
        return ret


class SweepIterator:
    """Iterator for the :class:`.Sweep` class.

    Manages the actual iteration of the pointer, and the execution of action
    functions. Manages and updates the state of the sweep.
    """

    def __init__(self, sweep: Sweep,
                 state: Dict[str, Any],
                 pass_kwargs=Dict[str, Any],
                 action_kwargs=Dict[str, Dict[str, Any]]):

        self.sweep = sweep
        self.state = state
        self.pass_kwargs = pass_kwargs
        self.action_kwargs = action_kwargs

        if isinstance(self.sweep.pointer, Sweep):
            self.pointer = iter(self.sweep.pointer)
        elif isinstance(self.sweep.pointer, collections.abc.Iterator):
            self.pointer = self.sweep.pointer
        elif isinstance(self.sweep.pointer, collections.abc.Iterable):
            self.pointer = iter(self.sweep.pointer)
        else:
            raise TypeError('pointer needs to be iterable.')

    def __next__(self):
        ret = {}
        next_point = next(self.pointer)
        if produces_record(self.sweep.pointer):
            ret.update(next_point)

        pass_args = []
        if self.sweep.pass_on_returns:
            if isinstance(next_point, (tuple, list)):
                if not self.sweep.pass_on_none:
                    pass_args = [r for r in next_point if r is not None]
                else:
                    pass_args = list(next_point)
            elif isinstance(next_point, dict):
                if not self.sweep.pass_on_none:
                    self.pass_kwargs.update({k: v for k, v in next_point.items()
                                             if v is not None})
                else:
                    self.pass_kwargs.update(next_point)
            else:
                if self.sweep.pass_on_none or next_point is not None:
                    pass_args.append(next_point)

        for a in self.sweep.actions:
            this_action_kwargs = {}
            if self.sweep.pass_on_returns:
                this_action_kwargs.update(self.pass_kwargs)
            this_action_kwargs.update(
                self.action_kwargs.get(a.__name__, {}))

            action_return = a(*pass_args, **this_action_kwargs)
            if produces_record(a):
                ret.update(action_return)

            # actions always return records, so no need to worry about args
            if not self.sweep.pass_on_none:
                self.pass_kwargs.update({k: v for k, v in action_return.items()
                                         if v is not None})
            else:
                self.pass_kwargs.update(action_return)

        if self.sweep.record_none is False:
            for k in list(ret.keys()):
                if ret[k] is None:
                    ret.pop(k)

        return ret


def append_sweeps(first: Sweep, second: Sweep) -> Sweep:
    """Append two sweeps.

    Iteration over the combined sweep will first complete the first sweep, then
    the second sweep.
    """
    both = IteratorToRecords(
        AppendSweeps(first, second),
        *combine_data_specs(*(list(first.get_data_specs())
                            + list(second.get_data_specs())))
    )
    sweep = Sweep(both)
    Sweep.link_sweep_properties(first, sweep)
    return sweep


def zip_sweeps(first: Sweep, second: Sweep) -> Sweep:
    """Zip two sweeps.

    Iteration over the combined sweep will elementwise advance the two sweeps
    together.
    """
    both = IteratorToRecords(
        ZipSweeps(first, second),
        *combine_data_specs(*(list(first.get_data_specs())
                            + list(second.get_data_specs())))
    )
    sweep = Sweep(both)
    Sweep.link_sweep_properties(first, sweep)
    return sweep


def nest_sweeps(outer: Sweep, inner: Sweep) -> Sweep:
    """Nest two sweeps.

    Iteration over the combined sweep will execute the full inner sweep for each
    iteration step of the outer sweep.
    """
    outer_specs = outer.get_data_specs()
    outer_indeps = [s.name for s in outer_specs if s.depends_on is None]

    inner_specs = [s.copy() for s in inner.get_data_specs()]
    for s in inner_specs:
        if s.depends_on is not None:
            s.depends_on = outer_indeps + s.depends_on

    nested = IteratorToRecords(
        NestSweeps(outer, inner),
        *combine_data_specs(*(list(outer_specs) + inner_specs))
    )
    sweep = Sweep(nested)
    Sweep.link_sweep_properties(outer, sweep)
    return sweep


class CombineSweeps:

    _operator_symbol = None

    def __init__(self, first: Sweep, second: Sweep):
        self.first = first
        self.second = second

    def __iter__(self):
        raise NotImplementedError

    def __repr__(self):
        ret = self.__class__.__name__ + ":\n"
        ret += indent_text(self.first.__repr__(), 4) + '\n'
        sym = ''
        if self._operator_symbol is not None:
            sym += self._operator_symbol + ' '
        sec_text = indent_text(self.second.__repr__(), 2)
        sec_text = sym + sec_text[len(sym):]
        ret += indent_text(sec_text, 4) + '\n'
        ret = ret.rstrip()
        while ret[-1] == "\n" and ret[-2] == "\n":
            ret = ret[:-1]
        return ret

    def get_data_specs(self):
        specs = list(self.first.get_data_specs()) + \
                list(self.second.get_data_specs())
        return combine_data_specs(*specs)


class ZipSweeps(CombineSweeps):

    _operator_symbol = '*'

    def __iter__(self):
        for fd, sd in zip(self.first, self.second):
            ret = fd.copy()
            ret.update(sd)
            yield ret


class AppendSweeps(CombineSweeps):

    _operator_symbol = '+'

    def __iter__(self):
        for ret in itertools.chain(self.first, self.second):
            yield ret


class NestSweeps(CombineSweeps):

    _operator_symbol = '@'

    def __iter__(self):
        for outer in self.first:
            for inner in self.second:
                ret = outer.copy()
                ret.update(inner)
                yield ret
