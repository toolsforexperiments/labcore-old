from typing import Any


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