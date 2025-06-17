#!/bin/sed -f

## Convert SMT-LIB psi file with neuron variables to a format suitable for QE

s/QF_LRA/LRA/

/\.frame/d

/^(declare-fun [^x]/{
    s/^(declare-fun \([^x][^ ]*\) () \(Real\))$/  (\1 \2)/
    H
    d
}

# Do this only for the first assert
1,/^(assert/{
    # Reset any previous substitution results
    t dummy
    : dummy
    s/(assert/&/
    # Do this only for the assert line, not the preceding ones
    T skip
    i(assert (exists (
    x
    p
    x
    i)
    i(and
    s/^(assert \(.*\))$/  \1/
    : skip
}

s/^(assert \(.*\))$/  \1/

/(check-sat/i)))
