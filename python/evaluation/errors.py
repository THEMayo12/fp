
import sys
import inspect

# ==================================================
#	error functions
# ==================================================


def format_type(expected_type):
    """format type expect they are tuple or pure types"""

    if isinstance(expected_type, tuple):
        expect = ["'{t}'".format(t=t.__name__) for t in expected_type]
        return ", ".join(expect)
    else:
        expect = expected_type.__name__
        return "'{t}'".format(t=expect)


def import_error(module):
    """exit if module was not found"""

    raise ImportError(
        "Can't find module {m}".format(m=module)
    )


def key_error(key):
    """exit if key was not found"""

    raise ImportError(
        "Error: key not recognized {k}".format(k=key)
    )


def assert_key(key, value, expected_type, array=None):
    """assertions for keywords"""

    if array:
        assert_key(key, value, expected_type)
        if value not in array:
            raise AttributeError(
                "Expected {k} to be on of {e}.\nGot '{g}'.".format(
                    k=key,
                    e=array,
                    g=value
                )
            )
    else:
        if not isinstance(value, expected_type):
            raise AttributeError(
                "Expected {k} to be {expect}. Got '{got}'.".format(
                    k=key,
                    expect=format_type(expected_type),
                    got=type(value).__name__
                )
            )


def assert_arg(value, expected_type):
    """assertions for arguments"""

    if not isinstance(value, expected_type):
        raise AttributeError(
            "Expected argument to be {expect}. Got '{got}'.".format(
                expect=format_type(expected_type),
                got=type(value).__name__
            )
        )
    else:
        pass


def assert_function(f):
    """assertions for functions"""
    if not inspect.isfunction(f):
        raise AttributeError("Expected argument to be a function.")


# ==================================================
#	main
# ==================================================


if __name__ == "__main__":

    # ===== import modules =============================

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.axes import Subplot

    # ===== test function ==============================

    def test(**keywords):
        for key in keywords:
            value = keywords[key]
            if key == "unit":
                assert_key(key, value, str)
            elif key == "prefix":
                assert_key(key, value, str, ["Hallo", "Du"])
            elif key == "name":
                assert_key(key, value, list)
            elif key == "sign":
                assert_key(key, value, (tuple, str))
            elif key == "np":
                assert_key(key, value, np.ndarray)
            else:
                key_error(key)


        return "Well done"

    # ===== test =======================================

    # print test(prefix="Hallou")
    # print test(np=np.array([1,2,3]))
    # print test(sign=[1,2])
    print test(np=True)
    # assert_arg(1, float)
    # def f(x): return x
    # assert_function(1.2)
