"""
REEDS
The aim of the module is to make the RE-EDS pipeline accesible to everyone! :)
"""

# Add imports here
import os, sys

##Import submodules
### PyGromos
sys.path.append(os.path.dirname(__file__)+"/submodules/pygromos")
import pygromos


from .reeds import *

# Handle versioneer
from ._version import get_versions


versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
