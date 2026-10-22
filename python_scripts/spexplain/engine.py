# Running the Explanation Computation Engine (the C++ executable `engine` inside of build*/)

import os
import subprocess
import sys
from pathlib import Path

from . import ROOT_DIR, Error

ENGINE_NAME = 'engine'
ENGINE_ENV_VAR = 'SPEXPLAIN_ENGINE'

# Networks in this format are processed by the frozen ONNX engine (see `make onnx`),
# which has the interface of the engine of that time
#+ temporary: integrate the ONNX engine into the main engine
# (e.g. NetworkOnnx, OpenSMTVerifierOnnx instead of Network2, OpenSMTVerifier2)
# and then drop the separate engine and the translation of its arguments (onnx_engine_args)
ONNX_SUFFIX = '.onnx'

# Short options of the frozen ONNX engine that take an argument
ONNX_ENGINE_SHORT_OPTS_WITH_ARG = 'VEesnit'


def uses_onnx(args):
    """Whether the engine arguments ('<action> <nn_model_fn> ...') refer to an ONNX network."""
    return len(args) >= 2 and Path(args[1]).suffix == ONNX_SUFFIX


def _onnx_engine_samples(spec):
    """Translate the samples '<idx>' or '[<idx>]-[<idx2>]' to the former '<idx>[,<idx2>]'."""
    first, dash, last = spec.partition('-')
    if not dash:
        return f'{first},{first}'
    return f'{first or 1},{last}' if last else first


def onnx_engine_args(args):
    """Translate the engine arguments to the interface of the frozen ONNX engine."""
    action, *rest = args
    positionals = []
    while rest and not rest[0].startswith('-') and len(positionals) < 3:
        positionals.append(rest.pop(0))
    # The former engine always takes the third positional argument as the strategies spec
    if action == 'explain' and len(positionals) == 2 and rest:
        positionals.append('')

    options = []
    has_format = False
    while rest:
        arg = rest.pop(0)
        if arg.startswith('--'):
            name, eq, value = arg.partition('=')
            name = name.replace('guide', 'prefer')
            if name == '--format':
                has_format = True
            elif name == '--samples':
                if not eq:
                    value = rest.pop(0) if rest else ''
                    eq = '='
                value = _onnx_engine_samples(value)
            options.append(name + eq + value)
        elif arg.startswith('-') and len(arg) > 1:
            # A bundle of short options, the last of them may take an argument
            for k, opt in enumerate(arg[1:], start=1):
                if opt in ONNX_ENGINE_SHORT_OPTS_WITH_ARG:
                    value = arg[k + 1:] or (rest.pop(0) if rest else '')
                    options += [f'-{opt}', _onnx_engine_samples(value) if opt == 'i' else value]
                    break
                if opt == 'B':
                    options.append('--format=bounds')
                    has_format = True
                    continue
                if opt in 'SI':
                    has_format = True
                options.append(f'-{opt}')
        else:
            options.append(arg)

    # The former default format was bounds
    if action == 'explain' and not has_format:
        options.append('--format=smtlib2')

    return [action, *positionals, *options]


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

    Priority: explicit path, $SPEXPLAIN_ENGINE, build-onnx/ for ONNX networks,
    build-marabou/ if Marabou is used, build/.
    """
    if engine_path is None:
        engine_path = os.environ.get(ENGINE_ENV_VAR)
    if engine_path is not None:
        path = Path(engine_path)
        hint = f'Check the --engine option or ${ENGINE_ENV_VAR}.'
    else:
        if uses_onnx(args):
            build_dir, make_target = 'build-onnx', 'make onnx'
        elif uses_marabou(args):
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
    path = find(args, engine_path)
    if uses_onnx(args):
        args = onnx_engine_args(args)
    cmd = [str(path), *args]
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
