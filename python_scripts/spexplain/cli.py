# Dispatching of the actions of spexplain

import sys

from . import Error, engine, explain

# Actions implemented directly by the engine, without any intervention
ENGINE_ACTIONS = ('dump-psi',)
ACTIONS = ('explain', *ENGINE_ACTIONS, 'interpret')
DEFAULT_ACTION = 'explain'


def print_rows(rows, file=sys.stdout):
    """Print rows of columns aligned with spaces."""
    widths = [max(len(row[i]) for row in rows if i < len(row) - 1) for i in range(max(map(len, rows)) - 1)]
    for row in rows:
        cells = [cell.ljust(widths[i]) for i, cell in enumerate(row[:-1])] + [row[-1]]
        print('    ' + '  '.join(cells), file=file)


def print_usage(file=sys.stdout):
    print('USAGE: spexplain [--engine <path>] [<action>] <args> [<options>]', file=file)
    print(f"ACTIONS: [{DEFAULT_ACTION}] {' '.join(a for a in ACTIONS if a != DEFAULT_ACTION)}", file=file)
    print_rows([
        ('explain:', '<nn_model_fn> <dataset_fn> [<exp_strategies_spec>]', 'Run the Explanation Computation Engine'),
        ('dump-psi:', '<nn_model_fn>', 'Print the SMT-LIB encoding of the NN'),
        ('interpret:', '[<explanation>...] <technique_spec> [<dataset>]', 'Run the Explanation Interpreter'),
    ], file=file)
    print("Use 'spexplain <action> --help' to print the usage of a particular action.", file=file)
    print('OPTIONS:', file=file)
    print_rows([
        ('--engine <path>',
         f'Use the given engine executable (default: ${engine.ENGINE_ENV_VAR} or build*/{engine.ENGINE_NAME})'),
    ], file=file)


def pop_engine_path(args):
    """Remove the wrapper option --engine from args (in place), return its value."""
    return explain.pop_long_opt(args, '--engine')


def run(args):
    if not args or args[0] in ('-h', '--help'):
        print_usage()
        return 0

    args = list(args)
    engine_path = pop_engine_path(args)

    if args and args[0] in ACTIONS:
        action, args = args[0], args[1:]
    else:
        action = DEFAULT_ACTION

    if action == 'explain':
        return explain.main(args, engine_path)
    if action in ENGINE_ACTIONS:
        if not args or args[0] in ('-h', '--help'):
            engine.print_usage(engine_path)
            return 0
        return engine.run([action, *args], engine_path)
    if action == 'interpret':
        from . import interpret
        return interpret.main(args)

    assert False, action


def main(args):
    try:
        return run(args)
    except Error as e:
        print(f'spexplain: error: {e}', file=sys.stderr)
        return 1
    except OSError as e:
        print(f'spexplain: error: {e.strerror}: {e.filename}', file=sys.stderr)
        return 1
