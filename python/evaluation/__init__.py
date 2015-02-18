
"""
                              _             _   _
              _____   ____ _| |_   _  __ _| |_(_) ___  _ __
             / _ \ \ / / _` | | | | |/ _` | __| |/ _ \| '_ \
            |  __/\ V / (_| | | |_| | (_| | |_| | (_) | | | |
             \___| \_/ \__,_|_|\__,_|\__,_|\__|_|\___/|_| |_|

"""

###############################################################################


import sys
import warnings
import errors as er

# import numpy
try:
    import numpy as np
except ImportError:
    er.import_error("numpy")

# import uncertainties
try:
    import uncertainties as uc
    import uncertainties.umath as um
except ImportError:
    er.import_error("uncertainties")

# for function plot_layout(ax)
try:
    from matplotlib.ticker import AutoMinorLocator
    from matplotlib.axes import Subplot
except ImportError:
    er.import_error("matplotlib")

# for fit function
try:
    from scipy import optimize
except ImportError:
    er.import_error("scipy")


###############################################################################


# Attributes that are always exported
__all__ = [
    "get_data",
    "get_uncert",
    "latexEq",
    "tex_eq",
    "tex_linreg"
    "plot_layout",
    "write"
]

###############################################################################

# ==================================================
#       settings
# ==================================================

_use_unitpkg = "siunitx"


def use_unitpkg(unit):
    """set the latex-package for units.

    Args:
        unit (str): the latex-package

    Returns:
        None

    """
    er.assert_arg(unit, str)
    if unit not in ["units", "siunitx"]:
        raise Exception(
            "Expected unit to be on of 'units', 'siunitx'. Got {g}".format(
                g=unit
            )
        )

    global _use_unitpkg
    _use_unitpkg = unit

# ==================================================
#       read data
# ==================================================


def get_data(file_name, **keywords):
    """get data from a textfile

    Args:
        file_name (str): filename as string

    Kwargs:
        index (int): if there are multiple tables in file, you can access them
            with index. The index starts with 0.

        unpack (bool): unpack table in such way to save a column
            to a variable.
            Example:
                a, b = get_data("example.txt", unpack=True)

        delimiter (str): columnseparator. Default: "\t"

        index_delim (str): separator between the tables in file.
            Default: "\n\n"

        usecols (list, tuple): array of column-numbers to get.
            Example: usecols=[1,3] of a table with ex. 4 columns picks the
                the first and third columns.

    Returns:
        ndarray.

    """

    # keywords
    index = 0
    unpack = False
    delimiter = '\t'
    index_delim = "\n\n"
    usecols = []

    # userdefined keywords
    for key in keywords:
        value = keywords[key]
        if key == "index":
            er.assert_key(key, value, int)
            index = value
        elif key == "unpack":
            er.assert_key(key, value, bool)
            unpack = value
        elif key == "delimiter":
            er.assert_key(key, value, str)
            delimiter = value
        elif key == "index_delim":
            er.assert_key(key, value, str)
            index_delim = value
        elif key == "usecols":
            er.assert_key(key, value(tuple, list))
            usecols = value
        else:
            er.key_error(key)

    # open file and read in
    File = open(file_name, "r")
    Return = [line for line in File if line[0] != "#"]

    temp = ""

    for line in Return:
        temp = "".join([temp, line])

    Return = temp.split("\n" + index_delim)

    while "" in Return:
        Return.remove("")

    Return = Return[index].split("\n")

    while "" in Return:
        Return.remove("")

    # convert string to float
    for i in range(len(Return)):
        Return[i] = Return[i].split(delimiter)
        Return[i] = [
            float(Return[i][j]) for j in range(len(Return[i]))
        ]

    # convert to numpy-array
    Return = np.array(Return)

    # evaluate keywords
    if len(usecols) > 0:
        assert len(Return[0] == len(usecols))
        Return = np.array([Return[:, col] for col in usecols])

    if unpack:
        Return = Return.transpose()

    if len(Return) == 1:
        if len(Return[0]) == 1:
            Return = Return[0][0]
        else:
            Return = np.array([i for i in Return[0]])

    return Return

# ==================================================
# 	functions for evaluation
# ==================================================


def get_uncert(x):
    """get uncertainty from array. -> calc mean and standard deviation.

    Args:
        x (list, ndarray): the array to calc the ufloat from.

    Returns:
        ufloat.

    """
    er.assert_arg(x, (list, np.ndarray))

    return uc.ufloat(np.mean(x), np.std(x))


def latexEq(x, **keywords):
    """function for writing latex-environments

    Args:
        x (str, list, tuple, ndarray): the equation-string('s)

    Kwargs:
        environment (string): the latex-environment .
        dot (bool): end of sentence after environment? -> dot?

    Returns:
        str. latex-environment

    Example:
        >>> eq = "c^2 = a^2 + c^2"
        >>> print latexEq([eq, eq], dot=True, environment="align")
        %
        \begin{align}
            c^2 &= a^2 + c^2        \\
            c^2 &= a^2 + c^2         ~.
        \end{align}
        %

    """

    # ===== variables ==================================

    # the array environments
    array_env = ["align", "aligned", "gather", "gathered", "eqnarray"]

    # single environments
    single_env = ["equation"]

    # the environment
    env = '$'

    # get environment without "*"
    pure_env = ""

    # the punctuation character
    dot = False

    # ===== keywords ===================================

    for key in keywords:
        value = keywords[key]
        if key == "environment":
            er.assert_key(key, value, str)
            er.assert_key(key, value.rstrip("*"), str, array_env + single_env)
            env = value
            pure_env = env.rstrip("*")
        elif key == "dot":
            er.assert_key(key, value, bool)
            dot = value
        else:
            er.key_error(key)

    # ===== check types ================================

    if pure_env in array_env:
        er.assert_arg(x, (list, tuple, np.ndarray))
        for string in x:
            er.assert_arg(string, str)
    else:
        er.assert_arg(x, str)

    # ===== create the environment =====================

    if env == '$':
        tex = x.join([env, env])
    else:
        # write the environment
        tex = "%\n\\begin{{{e}}}\n".format(e=env)

        # ===== align, aligned, ... ========================

        if pure_env in array_env:
            # the environment specific equality
            equal_sign = "="

            # the format of the row
            row = "\t{eq}\t\\\\\n"
            last_row = "\t{eq}\t"

            # assign the equality-sign
            if pure_env in ["align", "aligned"]:
                equal_sign = "&="
            elif pure_env in ["eqnarray"]:
                equal_sign = "&=&"
            else:
                pass

            # write the array
            for eq in x[:-1]:
                fixed_eq = eq.replace("=", equal_sign)
                filled_row = row.format(eq=fixed_eq)
                tex = "".join([tex, filled_row])

            # write the last element of the array
            fixed_eq = x[-1].replace("=", equal_sign)
            filled_row = last_row.format(eq=fixed_eq)
            tex = "".join([tex, filled_row])

        # ===== equation, ... ==============================

        else:
            tex = "".join([tex, "\t{eq}".format(eq=x)])

        # ===== set punctuation character ==================

        if dot:
            tex = "".join([tex, " ~.\n"])
        else:
            tex = "".join([tex, " \n"])

        tex = "".join([tex, "\\end{{{e}}}\n%".format(e=env)])

    return tex


def get_std(x):
    """get standard-deviations from covariance martix

    Args:
        x (list, ndarray): covariance martix

    Returns;
        ndarray. array with standard deviations.

    """
    er.assert_arg(x, (list, np.ndarray))
    return np.sqrt(np.diagonal(x))


def fit(f, x, y, p0=None, sigma=None, **keywords):
    """fit data with scipy.optimize.curvefit()"""
    er.assert_function(f)
    er.assert_arg(x, (list, np.ndarray))
    er.assert_arg(y, (list, np.ndarray))

    val, cov = optimize.curve_fit(f, x, y, p0, sigma, **keywords)
    return [val, get_std(cov)]

# ==================================================
# 	plot layout
# ==================================================


def plot_layout(ax):
    """default layout for plotting with matplotlib

    Args:
        ax (matplotlib.axes.Subplot): modify layout

    Returns:
        matplotlib.axes.Subplot.
    """
    er.assert_arg(ax, Subplot)

    ax.set_axisbelow(True)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.grid(which='major', axis='x',
            linewidth=0.75, linestyle='-', color='0.85')
    ax.grid(which='minor', axis='x',
            linewidth=0.25, linestyle='-', color='0.90')
    ax.grid(which='major', axis='y',
            linewidth=0.75, linestyle='-', color='0.85')
    ax.grid(which='minor', axis='y',
            linewidth=0.25, linestyle='-', color='0.90')

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    leg = ax.get_legend()
    if leg:
        ltext = leg.get_texts()
        frame = leg.get_frame()
        frame.set_facecolor('0.90')
        frame.set_edgecolor('0.90')

    return ax

# ==================================================
# 	format data for output
# ==================================================


def stretch_array(x, **keywords):
    """stretch and/or shift an array

    Args:
        x (list, ndarray): array to stretch

    Kwargs:
        factor (int): factor to stretch.
        shift (int): amount of times to shift.
        fill (str): fill character.

    Defaults:
        factor = 1
        shift = 0
        fill = ""

    Returns:
        list.

    Example:
        >>> a = [1, 2, 3, 4, 5]
        >>> stretch_array(a, factor=2, shift=1, fill="*")
        ['*', 1, '*', 2, '*', 3, '*', 4, '*', 5]
    """

    # define shift function
    def shift_list(L, n):
        n = n % len(L)
        return L[-n:] + L[:-n]

    factor = 1
    shift = 0
    fill = ""

    for key in keywords:
        value = keywords[key]
        if key == "factor":
            er.assert_key(key, value, int)
            factor = value
        elif key == "shift":
            er.assert_key(key, value, int)
            shift = value
        elif key == "fill":
            er.assert_key(key, value, str)
            fill = value
        else:
            er.key_error(key)

    L = []
    for i in range(len(x) * factor):
        if i % factor == 0:
            L.append(x[i/factor])
        else:
            L.append(fill)

    return shift_list(L, shift)

# ==================================================
# 	easy output function to write a string
# 	to a file
# ==================================================


def write(filename, string, option="w"):
    """write a string to a file

    Args:
        filename (str): filename with path
        string (str): the string
        option (str): the option to open file with builtin open()

    """

    if not filename.endswith('.tex'):
        filename = "{}.tex".format(filename)
    else:
        pass

    with open(filename, option) as f:
        if option == "a":
            f.write("\n\n")
        f.write(string)


def tex_eq(uc_val, **keywords):
    """ Create latex equation.

    Args:
        uc_val (uncertainties.Variable): the ufloat.

    Kwargs:
        name (str): name with variable in braces. Ex: "G(x)".
                    Default: None
        form (str): string of the format of the value with needed braces
                    for latex. Ex: "{:0.2f}". Default: "{:L}"
        unit (str): unit as string for latex. Ex: "\milli\second".
                    Default: ""

    """

    # ===== keywords ===================================

    unit = ""
    form = "{:L}"
    name = None

    # ===== check keywords =============================

    for key in keywords:
        value = keywords[key]
        if key == "unit":
            er.assert_key(key, value, str)
            unit = value
        elif key == "form":
            er.assert_key(key, value, str)
            form = value
        elif key == "name":
            er.assert_key(key, value, str)
            name = value
        else:
            er.key_error(key)

    # ===== add "=" to name ============================

    if name:
        funcname = "".join([name, " = "])
    else:
        funcname = ""

    if _use_unitpkg == "siunitx":
        eq = r"\SI[parse-numbers = false]{a}{b}"
        eq = eq.format(
            a="".join(["{{", form, "}}"]),
            b="{{{}}}",
            )
        eq = eq.format(
            uc_val,
            unit,
            )
    else:
        eq = r"\unit{a}{b}"
        eq = eq.format(
            a="".join(["[", form, "]"]),
            b="{{{}}}",
            )
        eq = eq.format(
            uc_val,
            unit
            )

    if name:
        return "".join([funcname, eq])
    else:
        return eq


def tex_linreg(name, val, std, **keywords):
    """ Create latex equation for linear regression.

    Args:
        name (str): name with varible in braces. Ex: "G(x)"
        val (list): the nominal values
        std (list): the errors of the nominal values

    Kwargs:
        form (list):    List with the format of the values with needed braces
                        for latex. Ex: "{:0.2f}"
        unit (list):    List with units as strings without needed braces
                        for latex. Ex: "\milli\second"


    """

    # ===== keywords ===================================

    unit = ["", ""]
    form = ["{:L}", "{:L}"]

    # ===== check keywords =============================

    for key in keywords:
        value = keywords[key]
        if key == "unit":
            er.assert_key(key, value, list)
            if len(value) != 2:
                raise Exception(
                    "Lenght of {} has to be 2, not {}!".format(key, len(value))
                )
            for item in value:
                er.assert_arg(item, str)
            unit = value
        elif key == "form":
            er.assert_key(key, value, list)
            if len(value) != 2:
                raise Exception(
                    "Lenght of {} has to be 2, not {}!".format(key, len(value))
                )
            for item in value:
                er.assert_arg(item, str)
            form = value
        else:
            er.key_error(key)

    # ===== define name and extract variable ===========

    funcname = "".join([name, " = "])
    variable = r'\, \cdot \,' \
        + funcname[funcname.find("(") + 1: funcname.rfind(")")] \
        + r'\,'

    # ===== create equation with siunitx or units ======

    if _use_unitpkg == "siunitx":
        eq = funcname + r"\SI[parse-numbers = false]{a}{b}" + variable \
            + r" {s} \SI[parse-numbers = false]{c}{d}"
        eq = eq.format(
            a="".join(["{{", form[0], "}}"]),
            b="{{{}}}",
            c="".join(["{{", form[1], "}}"]),
            d="{{{}}}",
            s="-" if val[1] < 0.0 else "+"
            )
        eq = eq.format(
            uc.ufloat(val[0], std[0]),
            unit[0],
            abs(uc.ufloat(val[1], std[1])),
            unit[1],
            )
    else:
        eq = funcname + r"\unit{a}{b}" + variable \
            + r" + \unit{c}{d}"
        eq = eq.format(
            a="".join(["[", form[0], "]"]),
            b="{{{}}}",
            c="".join(["[", form[1], "]"]),
            d="{{{}}}"
            )
        eq = eq.format(
            uc.ufloat(val[0], std[0]),
            unit[0],
            uc.ufloat(val[1], std[1]),
            unit[1],
            )

    return eq

# ==================================================
# 	main
# ==================================================


if __name__ == "__main__":
    a = np.array([1.1, 1.2, 0.9, .089, 1.33])

    u = get_uncert(a)
    print tex_eq(u, unit="\mA", form="({:L})")
