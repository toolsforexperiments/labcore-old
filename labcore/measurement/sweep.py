from typing import Iterable, Callable
import collections
import logging

from .record import produces_record, DataSpec, IteratorToRecords


logger = logging.getLogger(__name__)


def sweep_parameter(param: DataSpec, sweep_iterable: Iterable,
                    *actions: Callable):
    record_iterator = IteratorToRecords(sweep_iterable, param)
    return Sweep(record_iterator, *actions)


class Sweep:

    # TODO: action and pointer must not have conflicting data specs

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

        for a in self.sweep.actions:
            action_return = a(*args, **kwargs)
            if produces_record(a):
                ret.update(action_return)

        return ret
