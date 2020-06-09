from dataclasses import dataclass, field
from typing import Optional, Iterable, List, Callable, Iterator, Tuple, Union, \
    Any
from functools import update_wrapper
import copy
import collections
from enum import Enum
import logging

from ..utils import map_input_to_signature


logger = logging.getLogger(__name__)


class DataType(Enum):
    """Valid options for data types used in :class:`DataSpec`"""
    #: scalar (single-valued) data. typically numeric, but also bool, etc.
    scalar = 'scalar'
    #: multi-valued data. typically numpy-arrays.
    array = 'array'


@dataclass
class DataSpec:
    name: str
    depends_on: Union[None, List[str], Tuple[str]] = None
    type: Union[str, DataType] = 'scalar'
    unit: str = ''

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = DataType(self.type)

    def copy(self):
        """return a deep copy of the DataSpec instance."""
        return copy.deepcopy(self)


#: shorter notation for constructing DataSpec objects
ds = DataSpec
#: The type for creating a ds
DataSpecFromTupleType = Tuple[str, Union[None, List[str], Tuple[str]], str,
                              str]
#: Short hand for inputs we can make data specs from
DataSpecsType = Union[DataSpec, tuple, dict]


# TODO: just a string should be allowed
# TODO: make a document with an list of acceptable ways to specify ds
def make_data_specs(*specs: DataSpecsType):
    ret = []
    for spec in specs:
        if isinstance(spec, tuple):
            spec_ = DataSpec(*spec)
        elif isinstance(spec, dict):
            spec_ = DataSpec(**spec)
        elif isinstance(spec, DataSpec):
            spec_ = spec
        else:
            raise TypeError(f'specs must be either DataSpec, tuple, or dict,'
                            f'not {type(spec)}.')
        ret.append(spec_)
    ret = tuple(ret)
    return ret


def independent(name: str, unit: str = '', type: str = 'scalar') -> DataSpec:
    return DataSpec(name, unit=unit, type=type, depends_on=None)


indep = independent


def dependent(name: str, depends_on: List[str] = [], unit: str = "",
              type: str = 'scalar'):
    return DataSpec(name, unit=unit, type=type, depends_on=depends_on)


dep = dependent


def recording(*data_specs: [Union[DataSpec, tuple, dict]]):
    """Returns a decorator that allows adding data parameter specs to a
    function.
    """
    def decorator(func):
        return FunctionToRecords(func, data_specs)
    return decorator


def record_as(*args: Union[DataSpecsType, Callable, Iterable, Iterator]):
    if len(args) < 1:
        raise ValueError("need at least one object "
                         "producing output")
    obj = args[0]
    specs = make_data_specs(*args[1:])

    if isinstance(obj, Callable):
        return recording(*specs)(obj)
    elif isinstance(obj, collections.abc.Iterable):
        return IteratorToRecords(obj, specs)


def produces_record(obj):
    if hasattr(obj, 'get_data_specs'):
        return True
    else:
        return False


def _to_labelled_data(value, data_specs):
    ret = {}

    if isinstance(value, dict):
        for s in data_specs:
            ret[s.name] = value.get(s.name, None)

    elif isinstance(value, collections.abc.Iterator):
        ret = IteratorToRecords(value, data_specs)

    else:
        if not isinstance(value, (tuple, list)):
            value = [value]
        for i, s in enumerate(data_specs):
            try:
                ret[s.name] = value[i]
            except IndexError:
                ret[s.name] = None

    return ret


class IteratorToRecords:
    def __init__(self, iterable, data_specs):
        self.iterable = iterable
        if isinstance(data_specs, collections.abc.Iterable):
            self.data_specs = data_specs
        else:
            self.data_specs = make_data_specs(data_specs)

    def get_data_specs(self):
        return self.data_specs

    def __iter__(self):
        for val in self.iterable:
            yield _to_labelled_data(val, self.data_specs)


class FunctionToRecords:
    def __init__(self, func, data_specs):
        self.func = func
        if isinstance(data_specs, collections.abc.Iterable):
            self.data_specs = data_specs
        else:
            self.data_specs = make_data_specs(data_specs)
        update_wrapper(self, func)

    def get_data_specs(self):
        return self.data_specs

    def __call__(self, *args, **kwargs):
        func_args, func_kwargs = map_input_to_signature(self.func,
                                                        *args, **kwargs)
        ret = self.func(*func_args, **func_kwargs)
        return _to_labelled_data(ret, self.get_data_specs())