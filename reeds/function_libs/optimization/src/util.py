from typing import List


def read_gromos_csv(fileName, sep=' ', column=1):
    """
        @Old-deapreciated
        reads in a repdat file.

    Parameters
    ----------
    fileName: str
        path to the repdat file
    sep : str
        seperator char
    column : int


    Returns
    -------

    """

    data = []
    with open(fileName, 'r') as infile:
        for line in infile:
            words = list(filter(None, line.split(sep)))
            if len(words) > 0 and words[0].find('#') == -1:
                try:
                    data.append(float(words[column]))
                except IndexError:
                    print('WARNING: Line in %s contains too few coloumns: %s' % (fileName, line))
                except ValueError:
                    print('WARNING: "%s" in file %s cannot be converted to float.' % (words[column], fileName))
    return data


def get_str_from_list(in_list: List, val_type: str = '', n_per_line: int = 10) -> str:
    """
    Pretty print of list of values.

    Parameters
    ----------
    in_list : List
        List of values to print.
    val_type : str
        Print style of type. Possible values: "float", "int", ""
    n_per_line : int
        Maximum number of values per line

    Returns
    -------
    str
        message
    """

    # format function

    min_num = min(in_list)
    if ("e-" in str(min_num)):
        precision = int(str(min_num).split("e-")[1]) + 2
    else:
        precision = len(str(min_num))
    float_precision = ' {:8.' + str(precision) + 'f}'

    format_value = lambda val: str(val)
    if val_type == 'float':
        format_value = lambda val: float_precision.format(val)
    elif val_type == 'int':
        format_value = lambda val: ' {:8d}'.format(val)

    # print each line
    message = ""
    for line_id in range(int(len(in_list) / n_per_line) + 1):
        message += ''.join(format_value(val) for val in
                           in_list[line_id * n_per_line: min(len(in_list), (line_id + 1) * n_per_line)]) + "\n"

    return message
