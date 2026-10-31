"""Where the example scripts look for data.

The library itself needs no data -- every library function takes its paths as
parameters.  The *examples* read CSVs, ``.nnet`` models and ``.phi.txt``
explanation files, and those are too large to copy between projects.

The data root is resolved in this order, first hit wins:

1. ``$EXPLVIZ_DATA``, if set -- an explicit override, always respected.
2. ``<explviz>/../data``, if that directory exists -- i.e. a ``data/`` folder
   sitting next to the package.  This is anchored on the package's own location,
   not the working directory, so it works no matter where you run from::

       other-project/
           data/
               explanations/      <- .phi.txt files
               datasets/          <- .csv
               models/            <- .nnet
           explviz/               <- this package

3. The current directory, as a last resort.

So in a new project with a ``data/`` folder beside ``explviz/`` nothing needs to
be exported, and in this repo everything keeps working from the repo root.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Environment variable naming the directory that holds the data.
ENV_VAR = "EXPLVIZ_DATA"

#: This package's directory, used to anchor the ``../data`` lookup.
PACKAGE_DIR = Path(__file__).resolve().parent

#: Conventional data folder beside the package: ``<explviz>/../data``.
SIBLING_DATA_DIR = PACKAGE_DIR.parent / "data"

#: Subdirectory of the data root holding explanation ``.phi.txt`` files.
EXPLANATIONS_DIRNAME = "explanations"


def data_root() -> Path:
    """The directory example data paths are resolved against.

    Resolved on every call, so setting the environment variable or creating the
    sibling ``data/`` folder takes effect without reimporting.
    """
    configured = os.environ.get(ENV_VAR)
    if configured:
        return Path(configured)
    if SIBLING_DATA_DIR.is_dir():
        return SIBLING_DATA_DIR
    return Path(".")


def describe_root() -> str:
    """One line saying which root was chosen and why -- handy when a path fails."""
    configured = os.environ.get(ENV_VAR)
    if configured:
        return f"{ENV_VAR}={configured!r}"
    if SIBLING_DATA_DIR.is_dir():
        return f"the data/ folder beside the package ({SIBLING_DATA_DIR})"
    return f"the current directory ({Path.cwd()}), as no other root was found"


def data_path(*parts, must_exist: bool = True) -> Path:
    """Resolve ``parts`` against :func:`data_root`.

    Raises FileNotFoundError explaining which root was used and how to change
    it, rather than letting a bare path error surface from pandas or ``open()``.
    """
    path = data_root().joinpath(*parts)
    if must_exist and not path.exists():
        raise FileNotFoundError(
            f"explviz example data not found: {path}\n"
            f"  Data root is {describe_root()}.\n"
            f"  Either put a 'data/' folder next to the explviz package "
            f"({SIBLING_DATA_DIR}),\n"
            f"  or point the data root somewhere else:\n"
            f"    export {ENV_VAR}=/path/to/data"
        )
    return path


def explanations_path(*parts, must_exist: bool = True) -> Path:
    """Resolve ``parts`` against the ``explanations/`` folder of the data root."""
    return data_path(EXPLANATIONS_DIRNAME, *parts, must_exist=must_exist)
