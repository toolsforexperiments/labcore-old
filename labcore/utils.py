from typing import Any, Callable, Union
import inspect


def same_type(*args: Any, target_type: type = None) -> bool:
    """Check whether all elements of a sequence have the same type.

    :param seq: Sequence to inspect
    :param target_type: if not `None`, check if all elements are of that type.
        if `None`, only check if all elements are equal (of any type).
    :return: `True` if all equal, `False` otherwise.
    """

    if len(args) == 0:
        raise ValueError('nothing to compare. supply at least one argument.')

    for elt in args:
        if target_type is None:
            target_type = type(elt)

        if type(elt) is not target_type:
            return False

    return True


# FIXME: 'None' should never override a default!
def map_input_to_signature(func: Union[Callable, inspect.Signature],
                           *args: Any, **kwargs: Any):
    """Try to re-organize the positional arguments `args` and key word
    arguments `kwargs` such that `func` can be called with them.

    if `func` expects arguments that cannot be given, they will be given
    as ``None``.
    Surplus arguments are ignored if `func` does not accept variable positional
    and/or keyword arguments.
    Example::

        >>> def myfunc(x, y, z=1):
        ...     print(f"x={x}, y={y}, z={z}")
        ...
        ... args, kwargs = map_input_to_signature(myfunc, z=1, x=1, unused=4)
        ... myfunc(*args, **kwargs)
        x=1, y=None, z=1

    It is important to note that the position of positional arguments is not
    preserved, because input key words that match expected positional arguments
    are inserted as positional arguments at the right position. The order,
    however, is preserved. Example::

        >>> def myfunc(x, y, z):
        ...     print(f"x={x}, y={y}, z={z}")
        ...
        ... args, kwargs = map_input_to_signature(myfunc, 1, 2, x=5)
        ... myfunc(*args, **kwargs)
        x=5, y=1, z=2
    """
    args = list(args)
    func_args = []
    func_kwargs = {}

    if isinstance(func, inspect.Signature):
        sig = func
    else:
        sig = inspect.signature(func)

    # Logic:
    # for each param the function expects, we need to check if have
    # received a fitting one
    for idx, p in enumerate(sig.parameters):
        p_ = sig.parameters[p]
        if p_.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD,
                       inspect.Parameter.POSITIONAL_ONLY]:
            if p in kwargs:
                func_args.insert(idx, kwargs.pop(p))
            else:
                if len(args) > 0:
                    func_args.insert(idx, args.pop(0))
                elif p_.default is inspect.Parameter.empty:
                    func_args.insert(idx, None)
        elif p_.kind is inspect.Parameter.KEYWORD_ONLY:
            if p in kwargs:
                func_kwargs[p] = kwargs.pop(p)

        elif p_.kind is inspect.Parameter.VAR_POSITIONAL:
            for a in args:
                func_args.append(a)

        elif p_.kind is inspect.Parameter.VAR_KEYWORD:
            func_kwargs.update(kwargs)

    return func_args, func_kwargs
