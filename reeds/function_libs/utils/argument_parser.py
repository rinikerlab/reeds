"""
This file contains the dynamic argument parser used for many functions in the package.
The usecase for this is either in the bash command line function, or in the submission scripts.

"""
import argparse
import glob
import os
from inspect import signature, _empty
from typing import List, Tuple

from numpydoc.docscrape import NumpyDocString

from pygromos.euler_submissions import FileManager as fM
from reeds.function_libs.utils.structures import additional_argparse_argument


def make_parser_dynamic(parser: argparse.ArgumentParser, target_function: callable,
                        additional_argparse_argument: List[additional_argparse_argument],
                        function_parameter_exceptions: List[str],
                        verbose: bool = True) -> (dict, dict):
    """
        this function builds an argparse obj from a functions doc-string, this can be used to dynamicly parse the arguments for the function.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        parser to be build
    target_function : callable
        function to write the parser for
    additional_argparse_argument : List[additional_argparse_argument]
        additional provided parsing arguments? (eg. for single files)
    function_parameter_exceptions :  List[str]
        parameter not translated into the parser
    verbose : bool
        right now only dummy functionality

    Returns
    -------
    parsed arguments

    """
    function_signature = signature(target_function)
    function_doc = NumpyDocString(str(target_function.__doc__))

    # add arguments
    for (arg_name, arg_type, arg_required, arg_desc) in additional_argparse_argument:
        parser.add_argument('-' + arg_name, type=arg_type, required=arg_required, help=arg_desc)

    # get all args
    for param in function_signature.parameters:
        if (param in function_parameter_exceptions):
            continue
        else:

            parameter = function_signature.parameters[param]

            # add dynamicly help:
            try:
                param_desc = next(filter(lambda x: x.name == param, function_doc["Parameters"])).desc
            except Exception as err:
                print("\nCould not find Docstring for: ", parameter.name,
                      "\nPlease add it in the function declaration! and be careful about text formatting (not indent for docstrings)\n")

                print(err.args)
                param_desc = ""

            parser.add_argument('-' + parameter.name, type=parameter.annotation, default=parameter.default,
                                required=parameter.default == _empty,
                                help="\n".join(param_desc) + "\tdefault: [" + str(parameter.default) + "]")
    # user defined
    args, unkown_args = parser.parse_known_args()

    if (len(unkown_args) > 0):
        print("Input error! got an unkown argument: " + str(unkown_args))
        print()
        parser.print_help()
        args = -1

    return args


def execute_module_via_bash(module_doc: str, execute_function: callable,
                            requires_additional_args: List[Tuple[str, str]]):
    """
        This function checksout the provided do function of a module and builds an parser for bash
        from the provided doc-string (module_doc).

    Parameters
    ----------
    module_doc : str
        documentation of the module
    execute_function: callable
        do function of the module
    requires_additional_args: List[Tuple[str,str]
        additional required arguments, not in the docs?

    """
    from reeds.function_libs.utils.structures import additional_argparse_argument

    module_doc = NumpyDocString(str(module_doc))
    module_desc = "\n".join(module_doc['Summary']) + "\n" + "\n".join(module_doc['Extended Summary'])

    function_parameter_exceptions = ["in_SimSystem", "in_simSystem", "submit"]
    additional_argparse_arguments = [
        additional_argparse_argument(name='in_system_name', type=str, required=True, desc="give your system a name")]
    additional_argparse_arguments += [additional_argparse_argument(name=x[0], type=str, required=True, desc=x[1]) for x
                                      in requires_additional_args]

    parser = argparse.ArgumentParser(description=module_desc)
    parser.add_argument('--noSubmit', dest='submit', action='store_false',
                        help="Do not submit the jobs to the queue, just build structure")

    args = make_parser_dynamic(parser=parser, target_function=execute_function,
                               additional_argparse_argument=additional_argparse_arguments,
                               function_parameter_exceptions=function_parameter_exceptions, verbose=True)
    if (isinstance(args, int)):
        exit(1)

    # Build System:
    in_system_name = args.in_system_name
    in_topo_path = args.in_top_path
    in_coord_path = args.in_coord_path

    if (hasattr(args, "in_disres_path")):
        in_disres_path = args.in_disres_path
        del args["in_disres_path"]
    else:
        in_disres_path = None

    if (hasattr(args, "in_perttop_path")):
        in_perttopo_path = args.in_perttop_path
        del args["in_perttop_path"]
    else:
        in_perttopo_path = None

    in_cnf_path = in_coord_path if (os.path.isfile(in_coord_path)) else glob.glob(in_coord_path + "/*cnf")
    args = vars(args)

    del args["in_system_name"]
    del args["in_top_path"]
    del args["in_coord_path"]

    if ("exclude_residue" in args):
        args.update(
            {"exclude_residue": args.exclude_residue.split(" ") if (type(args.exclude_residue) != type(None)) else []})



    elif (isinstance(in_cnf_path, str) and not os.path.isfile(in_cnf_path)):
        print("ERROR - could not find Coordinate file: " + str(in_cnf_path))
        exit(1)

    top = fM.Topology(top_path=in_topo_path, disres_path=in_disres_path, pertubation_path=in_perttopo_path)
    system = fM.System(top=top, coordinates=in_coord_path, name=in_system_name)
    #args.update({"in_simSystem":system})

    # do everything in here :)
    ret = execute_function(in_simSystem=system, **args)

    # if error ocurred ret is smaller 0
    if (ret < 0):
        exit(1)
    else:
        exit(0)
