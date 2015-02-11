
"""
                 _                 _            _       _
             ___(_)_ __ ___  _ __ | | ___ _ __ | | ___ | |_
            / __| | '_ ` _ \| '_ \| |/ _ \ '_ \| |/ _ \| __|
            \__ \ | | | | | | |_) | |  __/ |_) | | (_) | |_
            |___/_|_| |_| |_| .__/|_|\___| .__/|_|\___/ \__|
                            |_|          |_|

"""


import sys
import inspect

###############################################################################


import errors as er

# import numpy
try:
    import numpy as np
except ImportError:
    er.import_error("numpy")

# for function plot_layout(ax)
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    from matplotlib.axes import Subplot
except ImportError:
    er.import_error("matplotlib")

###############################################################################


# Attributes that are always exported
__all__ = [
    "fig_size",
    "apply_params",
    "use_layoutfunc",
    "set_layoutfunc",
    "figure",
    "plot_layout"
]

###############################################################################

# ==================================================
# 	function to calc fig_size
# ==================================================


def fig_size(fig_width_pt):
    """calc figsize (golden mean) for matplotib's rcParams["figure.figsize"]

    Args:
        fig_width_pt (float): Get this from LaTeX using package printlen with
            \uselengthunit{pt}\printlength{\textwidth}

    Returns:
        list. [fig_width, fig_height]

    """

    # Convert pt to inch
    inches_per_pt = 1.0/72.27

    # Aesthetic ratio
    golden_mean = (np.sqrt(5) - 1.0)/2.0

    # width in inches
    fig_width = fig_width_pt*inches_per_pt

    # height in inches
    fig_height = fig_width*golden_mean

    return [fig_width, fig_height]


def golden_mean(width):
    """calc the height of a figure by given width

    Args:
        width (float): the width of figure

    Returns:
        float. the height of the figure with the golden mean
    """

    # Aesthetic ratio
    golden_mean = (np.sqrt(5) - 1.0)/2.0

    return width * golden_mean


# ==================================================
# 	settings
# ==================================================

# ===== some nice params for matplotlib ============

tex_preamble = [
    r"\usepackage{amsmath}",
    r"\usepackage[utf8]{inputenc}",
    r"\usepackage[T1]{fontenc}",
    r"\usepackage{upgreek}",
    r"\usepackage[nice]{units}",
    r"\usepackage{siunitx}",
]

tex_mtpro2_preamble = [
    r"\usepackage{amsmath}",
    r"\usepackage[utf8]{inputenc}",
    r"\usepackage[T1]{fontenc}",
    r"\usepackage{textcomp}",
    r"\renewcommand{\rmdefault}{ptm}",
    r"\usepackage{helvet}",
    r"\usepackage[subscriptcorrection,slantedGreek,nofontinfo]{mtpro2}",
    r"\usepackage{upgreek}",
    r"\usepackage[nice]{units}",
    r"\usepackage{siunitx}",
]


tex_mathdesign_preamble = [
    r"\usepackage{amsmath}",
    r"\usepackage[utf8]{inputenc}",
    r"\usepackage[T1]{fontenc}",
    r"\usepackage{txfonts}",
    r"\usepackage{times}",
    r"\usepackage[charter]{mathdesign}",
    r"\usepackage{upgreek}",
    r"\usepackage[nice]{units}",
    r"\usepackage{siunitx}",
]

tex_mathpazo_preamble = [
    r"\usepackage{amsmath}",
    r"\usepackage[utf8]{inputenc}",
    r"\usepackage[T1]{fontenc}",
    r"\usepackage[sc]{mathpazo}"
    r"\linespread{1.05}"
    r"\usepackage{upgreek}",
    r"\usepackage[nice]{units}",
    r"\usepackage{siunitx}",
]

tex_fouriernc_preamble = [
    r"\usepackage{amsmath}",
    r"\usepackage[utf8]{inputenc}",
    r"\usepackage[T1]{fontenc}",
    r"\usepackage{fouriernc}"
    r"\usepackage{newcent}"
    r"\usepackage{upgreek}",
    r"\usepackage[nice]{units}",
    r"\usepackage{siunitx}",
]

params = {
    'backend': 'pdf',
    'font.family': 'serif',
    'font.size': 10,
    'text.usetex': True,
    'text.latex.preamble': tex_preamble,
    'axes.labelsize': 10,
    # 'legend.fancybox': True,
    'legend.numpoints': 1,
    'legend.shadow': False,
    'legend.fontsize': 10,
    'xtick.direction': 'out',
    'xtick.labelsize': 10,
    'ytick.direction': 'out',
    'ytick.labelsize': 10,
    'figure.figsize': fig_size(418.25368),
    'axes.unicode_minus': True
}

# ===== settings for use of layout-function ========

# use this layout-function
_USE_LAYOUTFUNC = True

# the layout-function
_layoutfunc = None

# ==================================================
# 	function to update parameter and
# 	layoutfunction
# ==================================================


def apply_params():
    """sets the matplotlib default params and update this dictionary with
    the params dictionary.

    """
    mpl.rc_file_defaults()
    plt.rcParams.update(params)


def use_layoutfunc(u):
    """decide to use layoutfunction.

    Args:
        u (bool): set True to use layoutfunction.

    Returns:
        None.

    """
    er.assert_arg(u, bool)
    global _USE_LAYOUTFUNC
    _USE_LAYOUTFUNC = u


def set_layoutfunc(f):
    """set layoutfunction which is use in plot().

    Args:
        u (function): new layoutfunction.

    Returns:
        None.

    """
    if inspect.isfunction(f):
        global _layoutfunc
        _layoutfunc = f
    else:
        raise Exception(
            "Expected function."
        )


def set_fontsize(size):
    """set the fontsize of the plot. Everything will have that size
    (xticklabel, yticklabel, legend, axislabel...). Then the params
    will be updated.

    Args:
        size (int): the fontsize

    Returns:
        None.

    """
    params['text.fontsize'] = size
    params['axes.labelsize'] = size
    params['legend.fontsize'] = size
    params['xtick.labelsize'] = size
    params['ytick.labelsize'] = size

    plt.rcParams.update(params)


def show():
    """use matplotlibs show() function to show plot.
    If you want to save your plot, do not use this function before saving.

    """
    plt.show()

# ==================================================
# 	simpleplot
# ==================================================


class figure(object):
    """class for fast plotting with matplotlib"""

    def __init__(self, axis_label=None):
        """define the matplotlib figure"""

        # use default params
        plt.rcParams.update(params)

        # a list of dicts with {"x":_, "y":_, "keywords":_}
        self.__plots = []

        # list with x-, y-labels
        if axis_label:
            er.assert_arg(axis_label, (list, tuple))
            assert len(axis_label) == 2
            for item in axis_label:
                er.assert_arg(item, str)

            self.__xlabel = axis_label[0]
            self.__ylabel = axis_label[1]
        else:
            self.__xlabel = None
            self.__ylabel = None

        # check if legend is set
        self.__have_legend = False

        # xlim, ylim, samples
        self.__xlim = None
        self.__ylim = None
        self.__samples = 1000

    def clear_plots(self):
        self.__plots = []
        self.__have_legend = False
        self.__xlim = None
        self.__ylim = None
        self.__samples = 1000

    def set_xlabel(self, x):
        """set label of x-axis"""
        er.assert_arg(x, str)
        self.__xlabel = x

    def set_ylabel(self, y):
        """set label of y-axis"""
        er.assert_arg(y, str)
        self.__ylabel = y

    def set_samples(self, samples):
        """set number of values for all lineplots"""
        er.assert_arg(samples, int)
        self.__samples = samples

    def set_xlim(self, xlim):
        """set limit of x-axis"""
        er.assert_arg(xlim, list)
        if len(xlim) != 2:
            raise Exception(
                "Expected two itmes in list. Got {n}".format(n=len(xlim))
            )
        self.__xlim = xlim

    def set_ylim(self, ylim):
        """set limit of y-axis"""
        er.assert_arg(ylim, list)
        if len(ylim) != 2:
            raise Exception(
                "Expected two itmes in list. Got {n}".format(n=len(ylim))
            )
        self.__ylim = ylim

    def add_plot(self, *args, **keywords):
        """add plot to dictonary"""

        # ===== check args =================================

        isfunc = inspect.isfunction(args[0])
        if isfunc:
            for arg in args[1:]:
                er.assert_arg(arg, (int, float))
        else:
            if len(args) != 2:
                raise Exception(
                    "Expected two args in list. Got {n}".format(n=len(args))
                )
            er.assert_arg(args[0], (list, np.ndarray))
            er.assert_arg(args[1], (list, np.ndarray))

        # ===== check keywords =============================

        var_index = 0
        samples = 0

        for key in keywords:
            value = keywords[key]
            if key == "label":
                # er.assert_key(key, value, int)
                self.__have_legend = True
            elif key == "usevar":
                er.assert_key(key, value, (str, chr))
                if isfunc:
                    var_names = args[0].func_code.co_varnames
                    if value not in args[0].func_code.co_varnames:
                        raise Exception(
                            "Expected variable-name of function {f}.".format(
                                f=args[0].func_code.co_name)
                        )
                    var_index = var_names.index(value)
            elif key == "samples":
                er.assert_key(key, value, int)
                samples = value
            else:
                # let matplotlib do the work ;-)
                pass

        # delete key from keywords wich are not use in matplotlibs plot()
        keywords.pop("usevar", None)
        keywords.pop("samples", None)

        # ===== add to dictonary ===========================

        if isfunc:
            # default keywords
            kwrds = {"linestyle": "-"}
            kwrds.update(keywords)

            plot = {
                "x": None,
                "y": [args, var_index, samples],
                "keywords": kwrds
            }
        else:
            # default keywords
            kwrds = {"marker": "+", "linestyle": "none"}
            kwrds.update(keywords)

            plot = {
                "x": args[0],
                "y": args[1],
                "keywords": kwrds
            }

        self.__plots.append(plot)

    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # ===== fist plot values to get xlim ===============

        for plot in self.__plots:
            if plot["x"] is not None:
                ax.plot(
                    plot["x"],
                    plot["y"],
                    **plot["keywords"]
                )
            else:
                pass

        # ===== set limits =================================

        # for rouding errors
        eps = sys.float_info.epsilon

        # set xlim
        if self.__xlim:
            lim = self.__xlim
            ax.set_xlim(lim)
        else:
            # get xlim, default: [0, 1]
            lim = list(ax.get_xlim())

        # modify lim with eps to prevent rouding errors
        lim[0] += eps
        lim[1] -= eps

        # set ylim
        if self.__ylim:
            ax.set_ylim(self.__ylim)
        else:
            pass

        # ===== now plot functions =========================

        for plot in self.__plots:
            if plot["x"] is None:
                # function
                G = plot["y"][0][0]

                # arguments
                args = list(plot["y"][0][1:])

                # index of used varaible
                index = plot["y"][1]

                # samples for linsapce
                samples = plot["y"][2]

                # decide to use global or local samples (global in class)
                if samples:
                    pass
                else:
                    samples = self.__samples

                array = np.linspace(lim[0], lim[1], samples)
                args.insert(index, array)

                ax.plot(
                    array,
                    G(*args),
                    **plot["keywords"]
                )
            else:
                pass

        # ===== legend, labels, layout =====================

        # set labels
        if self.__xlabel:
            ax.set_xlabel(self.__xlabel)
        if self.__ylabel:
            ax.set_ylabel(self.__ylabel)

        # legend
        if self.__have_legend:
            ax.legend(loc='best')

        # layoutfunction
        if _USE_LAYOUTFUNC:
            if _layoutfunc:
                ax = _layoutfunc(ax)
            else:
                ax = plot_layout(ax)

        fig.tight_layout()

        # clear plots
        self.clear_plots()

        # return to save plots with fig.savefig(...)
        return fig

# ==================================================
# 	plot-layout
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
        # frame.set_facecolor((238/255., 233/255., 250/255.))
        # frame.set_edgecolor((238/255., 233/255., 250/255.))

    return ax

# ==================================================
# 	main
# ==================================================


def main():
    import evaluation.simpleplot as sp
    sp.params["text.latex.preamble"] = sp.tex_mtpro2_preamble

    # custom layoutfunction
    def layout(ax):
        leg = ax.get_legend()
        if leg:
            ltext = leg.get_texts()
            frame = leg.get_frame()
            # frame.set_facecolor('0.90')
            frame.set_facecolor((238/255., 233/255., 250/255.))
            frame.set_edgecolor((238/255., 233/255., 250/255.))

    # Geradenfunktion
    def G(x, m, b):
        return m*x + b

    # arrays
    t = np.array([1.2, 1.1, 1.3, 1.6, 1.7])
    s = t + 2

    p1 = sp.figure()
    p1.add_plot(t, s, label="$G(x)$")
    p1.add_plot(G, 2, 1, label="$G(x)$")
    p1.set_xlabel("Hallo")
    fig1 = p1.plot()
    # fig.savefig("test1.pdf")

    sp.set_layoutfunc(layout)
    sp.use_layoutfunc(False)

    p2 = sp.figure(["Hallo", "Du"])
    p2.set_xlim([0, 1])
    p2.add_plot(t, s, label="$G(x)$")
    p2.add_plot(G, 2, 1, usevar="m", label="$G(x)$", samples=10)
    fig1 = p2.plot()
    # fig.savefig("test1.pdf")

    sp.show()

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
