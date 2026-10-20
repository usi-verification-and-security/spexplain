# Running the Explanation Computation Engine (the C++ executable `engine` inside of build*/)

import os
import subprocess
import sys
from pathlib import Path

from . import ROOT_DIR, Error

ENGINE_NAME = 'engine'
ENGINE_ENV_VAR = 'SPEXPLAIN_ENGINE'


def uses_marabou(args):
    """Whether the engine arguments select the Marabou verifier."""
    for i, arg in enumerate(args):
        if arg in ('-V', '--verifier'):
            if i + 1 < len(args) and args[i + 1] == 'marabou':
                return True
        elif arg in ('-Vmarabou', '--verifier=marabou'):
            return True
    return False


def find(args, engine_path=None):
    """Locate the engine executable.

    Priority: explicit path, $SPEXPLAIN_ENGINE, build-marabou/ if Marabou is used, build/.
    """
    if engine_path is None:
        engine_path = os.environ.get(ENGINE_ENV_VAR)
    if engine_path is not None:
        path = Path(engine_path)
        hint = f'Check the --engine option or ${ENGINE_ENV_VAR}.'
    else:
        if uses_marabou(args):
            build_dir, make_target = 'build-marabou', 'make marabou'
        else:
            build_dir, make_target = 'build', 'make'
        path = ROOT_DIR / build_dir / ENGINE_NAME
        hint = f"Build it first with '{make_target}' in {ROOT_DIR}."

    if not (path.is_file() and os.access(path, os.X_OK)):
        raise Error(f'Engine executable not found: {path}\n{hint}')

    return path


def run(args, engine_path=None):
    """Run the engine with the arguments and return its exit code."""
    cmd = [str(find(args, engine_path)), *args]
    try:
        returncode = subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        return 130

    # Terminated by a signal
    if returncode < 0:
        return 128 - returncode
    return returncode


def print_usage(engine_path=None, file=sys.stdout):
    """Print the usage of the engine (it prints it when run without arguments)."""
    file.flush()
    out = subprocess.run([str(find([], engine_path))], stdout=subprocess.PIPE, text=True).stdout
    print(out, end='', file=file)
