import functools
import traceback


def print_exc(function):
    """
    A decorator that wraps the passed in function and prints any exceptions.
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except:
            traceback.print_exc()
            raise
    return wrapper
