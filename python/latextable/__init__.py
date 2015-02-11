
"""
                 _       _            _        _     _
                | | __ _| |_ _____  _| |_ __ _| |__ | | ___
                | |/ _` | __/ _ \ \/ / __/ _` | '_ \| |/ _ \
                | | (_| | ||  __/>  <| || (_| | |_) | |  __/
                |_|\__,_|\__\___/_/\_\\__\__,_|_.__/|_|\___|

"""
###############################################################################


import sys
import os.path

###############################################################################


__all__ = ["latextable"]
__author__ = ["Mario Dunsch"]

# ==================================================
#       check arguments of latextabel function
# ==================================================


def check_arg(_mat):
    """Checks the type of given matrix and convert to pure nested lists.

    Args:
        _mat (list, numpy.ndarray): Matrix which should formed to latex table

    Returns: list. Matrix as nested list.

    """
    try:
        mat = list(_mat)
    except TypeError, e:
        raise TypeError("Matrix has to be iterable: {}".format(e))
    except Exception, e:
        # maybe some unexpected errors
        raise e

    for i in range(len(mat)):
        try:
            mat[i] = list(mat[i])
        except TypeError, e:
            raise TypeError(
                "Rows and colums of the "
                "matrix hast to be iterable: {}".format(e)
            )
        except Exception, e:
            raise e

    return mat


def fill_empty(mat, fill):
    """Fill empty entries of matrix withs string.

    Args:
        mat (list): Matrix to fill entries with
        fill (str): String with which the empty entries will be filled.

    Returns: (int, int, list). Dimensione m, n and the matrix itself.

    """
    m, n = check_dim(mat)
    # emptystr: if rows has different dimensions, fill up list with emptystr
    for i in range(m):
        dn = n - len(mat[i])
        for j in range(dn):
            mat[i].append(fill)

    return m, n, mat


def check_dim(mat):
    """Check dimensions of the matrix.

    Args:
        mat (list): Matrix to check the dimension.

    Returns: (int, int). Dimensions m, n.

    """
    # rows
    m = len(mat)
    # columns
    n = 0

    # get the maximum length of columns
    for i in range(m):
        n_temp = len(mat[i])
        if n_temp > n:
            n = len(mat[i])

    return m, n


# ==================================================
#       helper function to format values
# ==================================================


def format_col(col, form):
    """Format a column with format strings.

    Args:
        col (list): A column of the matrix
        form (str): The format string

    Returns: str. Formatted entries as strings.

    """
    if isinstance(form, str):
        formatted_col = []

        for c in col:
            try:
                formatted_col.append(format(c, form))
            except ValueError, e:
                if isinstance(c, str):
                    formatted_col.append(c)
                else:
                    raise Exception("Format specifier hast wrong form!")

    elif isinstance(form, list):
        if len(col) != len(form):
            raise Exception(
                "List in keyword form doesn't match"
                "the dimension of the matrix!"
            )
        else:
            formatted_col = []
            for i, c in enumerate(col):
                try:
                    formatted_col.append(format(c, form[i]))
                except IndexError, e:
                    # here its a "real" index error
                    raise e
                except ValueError, e:
                    if isinstance(c, str):
                        formatted_col.append(c)
                    else:
                        raise Exception("Format specifier hast wrong form!")
    else:
        raise Exception("Keyword form has wrong type!")

    return formatted_col


# ==================================================
#       helper function to fing pre and post decimal
#       digits
# ==================================================


def get_digit_position(align_str):
    """Get the numbers of columns wich should be aligned at the dot.

    Args:
        align_str (str): The align string like "cclr"

    Returns: Reduced align string.

    """
    temp = list(align_str.replace('|', ''))
    return [i for i, val in enumerate(list(temp)) if val in ["C", "L", "R"]]


def get_digits(value):
    """Get the pre and post decimal digits of a float.

    Args:
        value (str): A formatted value

    Returns: tuple(tuple, bool). ((digits pre dot, digits post dot), has dot)

    """
    try:
        float(value)
    except ValueError, e:
        # None will be ignored of functions like max
        return (None, None), None

    pos = value.find(".")
    digits = [0, 0]

    if pos > -1:
        digits[0] = pos
        digits[1] = len(value) - (pos + 1)
        has_dot = True
    else:
        digits[0] = len(value)
        has_dot = False

    return tuple(digits), has_dot


def dot_in_col(col):
    """Determine, if value in column has dot respectivly is a float.

    Args:
        col (list): A column of the matrix

    Returns: list. List with bool, which denote which entrie of the column
        has a dot.

    """
    dots = []

    for i in col:
        try:
            float(i)
            if i.find("."):
                dots.append(True)
            else:
                dots.append(False)
        except ValueError, e:
            # None will be ignored of functions like max
            dots.append(False)

    return dots


def get_maxdigits(col):
    """Get the maximal pre and post decimal digits of a column.

    Args:
        col (list): A column of the matrix

    Returns: (maximal digits pre dot, maximal digits post dot).

    """
    digits = [get_digits(d)[0] for d in col]
    digits = zip(*digits)

    return tuple([max(digits[0]), max(digits[1])])


def align_value(value, max_digits, dot=True):
    """Align the value with resprect to the maximal pre and post digits.

    Args:
        value (str): A formatted value
        max_digits (tuple): maximal digits pre and post dot.
        dot (bool): Denotes, that there is a value with dot in the appropriate
            column.

    Returns: str. Value formatted for table with dot alignment.

    """
    digits, has_dot = get_digits(value)

    if None in digits:
        return value
    else:
        pre = ""
        post = ""

        diff_digits = (max_digits[0] - digits[0], max_digits[1] - digits[1])

        if diff_digits[0]:
            pre = r"\phantom{{{}}}".format(diff_digits[0]*"0")
        if diff_digits[1]:
            post = r"\phantom{{{}}}".format(diff_digits[1]*"0")

        # convert minus to dash for better alignment
        val = value.replace("-", "--")

        if dot and not has_dot:
            return "{a}{b}{c}{d}".format(a=pre, b=val, c="\phantom{.}", d=post)
        else:
            return "{a}{b}{c}".format(a=pre, b=val, c=post)


# ==================================================
#       function to split matrix
# ==================================================


def split(matrix, part, fill="-"):
    """Split matrix in 'part' parts and fill up empty entries with
    string 'fill' for the case the number of entries doesn't match to the parts
    to split.

    Args:
        matrix (list): matrix to split
        part (int): number of parts in which the matrix should be splitted
        fill (str): String to fill the empty entries

    Returns: list. The new splitted matrix.

    """
    # first get dimenstions an fill up empty entries
    m, n, mat = fill_empty(matrix, fill)
    # transpose for better use
    mat = [list(i) for i in zip(*matrix)]
    temp = m
    m = n
    n = temp

    # check if part is higher then the number of columns
    if part > n:
        raise Exception(
            "Can't split matrix in more parts then the number of columns.\n"
            "Columns: {}, Split: {}".format(n, part)
        )

    # define helper
    temp_mat = []
    new_mat = []

    # get the size of the splitted parts
    if n % part:
        chunk = (n / part) + 1
    else:
        chunk = (n / part)

    # split each column in blocks of size chunk and store in temp_mat
    for i, col in enumerate(mat):
        temp_mat.append([col[i:i+chunk] for i in range(0, len(col), chunk)])
    # fill up the temporary matrix
    for i, col in enumerate(temp_mat):
        fill_empty(col, fill)
    # [(a), (b)]
    # |(c), (d)| --> [(a), (b), (c), (d), (e), (f)] ,
    # [(e), (f)]
    # a, b, ... are the splitted parts in temp_mat.
    for i in range(part):
        # m = len(temp_mat) = len(mat)
        for j in range(m):
            new_mat.append(temp_mat[j][i])

    return [list(i) for i in zip(*new_mat)]


###########################################################################
#                               latextable                                #
###########################################################################


def latextable(matrix, filename=None, **keywords):
    """Convert arrays/matrices to latex-code

    Args:
        matrix (list, numpy.ndarray): Matrix to convert

        filename (str): File to place output.
            Extension .tex is added automatically.
            File can be included in a laTex document by \input{filename}.
            Output will always be returned in a string.
            Filename must be a string or None.
            If filename is None, no outputfile will be generated.
    Kwargs:
        alignment (str): A string of latex alignment like 'clr' and
            uppercase 'CLR'.
            The lenght of the string has to be equal to the number of columns.
            The uppercase characters are for interpretation
            as "dot-alignment" and will be rewritten as 'c'. To achieve
            dot-alignment, the latex-command '\phantom{...}' will appear
            with the required number of '0's to reserve the right
            space. Have a look at the example for better understanding
            of the idea.
            (See Keyword 'split' for additional use with split)

        transpose (bool): Transpose the input matrix, True.

        emptystr (str): String wich will be placed, if rows or columns doesn't
            have the full dimension, to get a full m x n matrix.

        form (list, str): List of strings or a string of the form '.2f'.
            String will extended internally to '{:.2f}' for use with builtin
            function 'format'. So you can use any format specifier.

            If 'form' is a list, every entry will be applied to the
            appropriate column.
            It's also possible to have a list of strings in that list, to
            format every entrie of an column different.
            If 'form' is a str, then this format will be applied to every
            column and every entrie.
            (See Keyword 'split' for additional use with split)

        split (int): If columns of a table are to long, you can split up the
            table into 'split' parts.
            If 'split' is set, the lenght of the string of the
            'alignment'-keyword and the list of the 'form'-keyword can have
            ('split') * (number of columns) for additional customization.

    Returns: str. Latex formated table.

    """

    # first check arg
    mat = check_arg(matrix)

    # define keywords
    align = "C"
    transpose = True
    emptystr = "-"
    form = ""
    split_parts = None

    # ===== check keyword args =========================

    for key in keywords:
        value = keywords[key]
        if key == "alignment":
            if not isinstance(value, str):
                raise TypeError(
                    "Keyword align has to be bool. Got {}".format(
                        type(value)
                    )
                )
            align = value
        elif key == "transpose":
            if not isinstance(value, bool):
                raise TypeError(
                    "Keyword transpose has to be bool. Got {}".format(
                        type(value)
                    )
                )
            transpose = value
        elif key == "emptystr":
            if not isinstance(value, str):
                raise TypeError(
                    "Keyword emptystr has to be a str. Got {}".format(
                        type(value)
                    )
                )
            emptystr = value
        elif key == "form":
            # check type later
            form = value
        elif key == "split":
            if not isinstance(value, int):
                raise TypeError(
                    "Keyword split has to be an int. Got {}".format(
                        type(value)
                    )
                )
            if value <= 0:
                raise Exception(
                    "Value of keyword split is not accepted. Got {}\n"
                    "Should be higher than 0.".format(value)
                )
            split_parts = value
        else:
            raise Exception("Key not recognized '{}'".format(key))

    # ===== now check dimensions of matrix =============

    m, n, mat = fill_empty(mat, emptystr)

    # ===== check some keywords ========================

    # transpose
    if transpose:
        mat = [list(i) for i in zip(*mat)]
        # swap dimensions
        temp = m
        m = n
        n = temp

    # now check and update align string
    if not set(align).issubset(set("crlCRL|")):
        raise Exception(
            "Characters of keyword 'align' should be one of 'crlCRL|'!"
        )
    elif len(align) == 1:
        align = align * n
        if split_parts:
            align = split_parts * align
        else:
            pass
    elif len(align.replace("|", "")) not in [n, n * split_parts]:
        raise Exception(
            "Dimensions of matrix and align string are different!"
        )
    elif split_parts:  # adjust the alignment sting, if split is set.
        align = split_parts * align
    else:
        pass

    # add extension if not present
    if filename and not filename.endswith(".tex"):
        filename = "{}.tex".format(filename)

    # check lenght of form string
    if isinstance(form, str) or len(form) not in [n, n * split_parts]:
        raise Exception(
            "Lenght of keyword 'form' doesn't match the number of columns."
        )
    elif split_parts:  # adjust the form list, if split is set.
        form = split_parts * form
    else:
        pass

    # ===== split matrix if key is set =================

    if split_parts:
        mat = split(mat, split_parts, emptystr)
        m, n = check_dim(mat)

    # ===== format elements ============================

    temp_mat = [list(i) for i in zip(*mat)]

    try:
        if isinstance(form, str):
            for i, col in enumerate(temp_mat):
                temp_mat[i] = format_col(col, form)
        else:
            for i, col in enumerate(temp_mat):
                temp_mat[i] = format_col(col, form[i])
    except IndexError, e:
        raise e

    mat = [list(i) for i in zip(*temp_mat)]

    # ===== align the values ===========================

    temp_mat = [list(i) for i in zip(*mat)]
    aligned_positions = get_digit_position(align)

    for i, col in enumerate(temp_mat):
        if i in aligned_positions:
            max_digits = get_maxdigits(col)
            dots = dot_in_col(col)
            for index, value in enumerate(col):
                col[index] = align_value(value, max_digits, dots[index])

    mat = [list(i) for i in zip(*temp_mat)]

    # ===== matrix to latex format =====================

    mat_list = []
    for row in mat:
        mat_list.append(" & ".join([s.ljust(17) for s in row]))
    mat_str = "\\\\\n".join(mat_list)

    # add a last "\\" to matrix string
    mat_str = "".join([mat_str, "\\\\"])

    # ===== write to file ==============================

    if filename:
        with open(filename, "w") as f:
            f.write(mat_str)

    # ===== ready to return ============================

    return mat_str


# ==================================================
#       main
# ==================================================


def main():
    # locale.setlocale(locale.LC_ALL, "de_DE")

    m1 = [[1, 2, 3], [5, 6]]
    m2 = [[107, 108, 111, 23, 0.2], [43.9, 33.2, 55.3, 12.33, 222.3]]

    print latextable(
        m2,
        # "hallo",
        alignment="CCcC",
        # form=[".0f", 3*["0.1f"] + ["0.2f"]]
        form=[".0f", ".3f"],
        split=2
        # form="0.2f"
    )
    # print latextable(m1, alignment="cC", form=[".2f", [".4f", ".1f", ".3f"]])
    # print get_digits("11.2")
    # print get_maxdigits(["1.2", "Hallodfjk", "22.222", "-234.2"])
    # print get_digits("-2.34")
    # print align_value("-2.34", (3,3))
    # print get_digit_position("ccC|CcC")
    # print format_col([1,2,3,"h"], ".2f")
    # print format_col([1,2,3,"h"], 4*["0.3f"])

    # print split(
    #     [
    #         [1,2,3],
    #         [3,4,3],
    #         [5,6,4],
    #         [7,8,5],
    #         [9,3,6],
    #         [4,3,7],
    #         [6,5,8],
    #         [9,3,0],
    #         [9,3,0]
    #     ],
    #     2,
    #     "-"
    # )

if __name__ == '__main__':
    sys.exit(main())
