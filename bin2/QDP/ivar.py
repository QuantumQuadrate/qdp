from numpy import *  # python35 cant have star import inside a function


def is_iterated(func):
    """Evaluates independent variables to see if it is a list(-ish) type.

    Need to have numpy functions defined in the namespace for CsPy, but we dont
    actually want to pollute the namespace.
    """
    try:
        tmp = eval(func.value)
    except NameError as e:
        print(e)
        return False
    if (type(tmp) == list) | (type(tmp) == ndarray) | (type(tmp) == tuple):
        return len(tmp) > 1
    return False
