import itertools
from typing import Iterable, Callable, Union, Tuple, Any
import collections
import logging

from .record import produces_record, DataSpec, IteratorToRecords, \
    DataSpecFromTupleType


logger = logging.getLogger(__name__)


null_pointer = [None]


def null_action():
    return None


def sweep_parameter(param: Union[str, DataSpecFromTupleType, DataSpec],
                    sweep_iterable: Iterable,
                    *actions: Callable):

    if isinstance(param, str):
        param_ds = DataSpec(param)
    elif isinstance(param, tuple):
        param_ds = DataSpec(*param)
    elif isinstance(param, DataSpec):
        param_ds = param

    record_iterator = IteratorToRecords(sweep_iterable, param_ds)
    return Sweep(record_iterator, *actions)


class Sweep:

    # TODO: action and pointer must not have conflicting data specs
    # TODO: allow for passing arguments to various functions
    # TODO: '*' should be zip
    #   '@' is probably a good way for out products

    def __init__(self, pointer: Iterable, *actions: Callable):
        if isinstance(pointer, (collections.abc.Iterable, Sweep)):
            self.pointer = pointer
        else:
            raise TypeError('pointer needs to be iterable.')

        self.actions = []
        for a in actions:
            if callable(a):
                self.actions.append(a)
            else:
                raise TypeError('action must be a callable.')

    def __iter__(self):
        return SweepIterator(self)

    # FIXME: i think this is not logical right now.
    #   if a callable is added, it should be executed once at the end.
    #   '*' or '@' would be the right thing if you want it executed always.
    def __add__(self, other: Union[Callable, "Sweep"]) -> "Sweep":
        if isinstance(other, Sweep):
            return append(self, other)
        elif callable(other):
            self.actions.append(other)
            return self
        else:
            raise TypeError(f'can only add Sweep or callable, '
                            f'not {type(other)}')

    def run(self):
        return SweepIterator(self)

    def get_data_specs(self):
        specs = []
        pointer_specs = []
        if produces_record(self.pointer):
            pointer_specs = self.pointer.get_data_specs()
            specs += pointer_specs
        for a in self.actions:
            if produces_record(a):
                action_specs = a.get_data_specs()
                pointer_independents = [ds.name for ds in pointer_specs
                                        if ds.depends_on is None]
                for aspec in action_specs:
                    aspec_ = aspec.copy()
                    if aspec_.depends_on is not None:
                        aspec_.depends_on = pointer_independents + aspec_.depends_on

                    specs.append(aspec_)

        return specs


class SweepIterator:

    def __init__(self, sweep: Sweep):
        self.sweep = sweep

        if isinstance(self.sweep.pointer, Sweep):
            self.pointer = iter(self.sweep.pointer)
        elif isinstance(self.sweep.pointer, collections.abc.Iterator):
            self.pointer = self.sweep.pointer
        elif isinstance(self.sweep.pointer, collections.abc.Iterable):
            self.pointer = iter(self.sweep.pointer)
        else:
            raise TypeError('pointer needs to be iterable.')

    def __iter__(self):
        return self

    def __next__(self):
        ret = {}
        next_point = next(self.pointer)
        if produces_record(self.sweep.pointer):
            ret.update(next_point)

        args = []
        kwargs = dict()
        if isinstance(next_point, (tuple, list)):
            args = next_point
        elif isinstance(next_point, dict):
            kwargs = next_point
        else:
            args.append(next_point)

        # TODO: would probably be good to check about passing arguments here.
        #   otherwise it'll only work for recording functions
        for a in self.sweep.actions:
            action_return = a(*args, **kwargs)
            if produces_record(a):
                ret.update(action_return)

        return ret


def append(first: Sweep, second: Sweep):
    both = IteratorToRecords(
        itertools.chain(first, second),
        first.get_data_specs() + second.get_data_specs()
    )
    return Sweep(both)
