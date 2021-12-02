import time
from functools import wraps


def timing(f):
    """Decorator for timing functions

    Args:
        f (function): function for decorating
        
    Usage:
    @timing
    def function(a):
        pass
    """
    
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f"function:{f.__name__} took: {end-start:2.5f}")
        return result
    
    return wrapper
