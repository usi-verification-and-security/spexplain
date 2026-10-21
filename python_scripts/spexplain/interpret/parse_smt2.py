"""
Basic module to parse an smt2 file which contains a conjunction of inequalities into a numpy array. The output 
of the functions interpret or parse_and_interpret is a numpy array of shape C x (N+1), where C is the number
of inequalities, and each row r = (r_1, ..., r_{N+1}) encodes an inequality: x_1 r_1 + ... + x_N r_N + r_{N+1} >= 0.
"""

import numpy as np
import pyparsing as pp

# Recursively interpret a list l of parsed linear terms in variables v, returns an array of shape (len(v) + 1), where 
# the last element is the constant term, and the array encodes the linear combination in l, i.e.
# r = _interpret_linear_term(l,v), then l is equivalent to sum_i r_i v_i, where the last element of v is the constant.
def _interpret_linear_term(l,v):
    def _is_const(t):
        return np.all(t[:-1] == 0.0)

    if type(l) is not list:
        # a variable, or a stringified float
        try:
            varpos = v.index(l)
            res = np.zeros(len(v) + 1)
            res[varpos] = 1.0
            return res
        except ValueError:
            val = float(l)
            res = np.zeros(len(v) + 1)
            res[-1] = val
            return res
    else:
        assert type(l) is list, '%s is not a list' % str(l)
        assert len(l) >= 2
        term1 = _interpret_linear_term(l[1],v)
        if len(l) == 3:
            term2 = _interpret_linear_term(l[2],v)
        elif len(l) == 2:
            term2 = None #special case: unary -
        elif len(l) > 3:
            term2 = None
            assert l[0] == '+'
            other_terms = [ _interpret_linear_term(l[p],v) for p in range(2,len(l)) ]

        if l[0] == '+':
            return term1 + term2 if term2 is not None else term1 + np.sum(other_terms,0)
        elif l[0] == '-':
            # term2 may not exist
            if term2 is None:
                return -term1
            else:
                return term1 - term2
        elif l[0] == '*':
            # term1 or term2 should be a constant
            if _is_const(term1):
                return term2 * term1[-1]
            elif _is_const(term2):
                return term1 * term2[-1]
            else:
                raise ValueError('%s is a nonlinear expression' % str(l))
        elif l[0] == '/':
            # term2 should be a constant
            if _is_const(term2):
                return term1 / term2[-1]
            else:
                raise ValueError('%s is a nonlinear expression in division' % str(l))
        else:
            raise ValueError('Cannot interpret linear term %s' % str(l))

# Interpret a parsed inequality of the form ('<=',t1,t2), (not ('<=' ...)), ('=' ...), ('>=' ...). If ('and' ...) is encountered,
# recurses into the tail of the expression.
# Returns a list of arrays of shape (len(v) + 1), where the last element is the constant term, and the array encodes the inequality in l, i.e.
# r = _interpret_inequality(l,v), then r = t2-t1, and the inequality is r >= 0. (In case of equality, a list of two opposing
# inequalities is returned)
def _interpret_inequality(l,v):
    if l[0] == 'and':
        res = [ _interpret_inequality(ineq,v) for ineq in l[1:] ]
        return sum(res,[]) # must flatten one level
    elif l[0] == 'not':
        assert len(l) == 2
        return [-x for x in _interpret_inequality(l[1],v)]
    elif l[0] == '<=':
        assert len(l) == 3
        t1 = _interpret_linear_term(l[1],v)
        t2 = _interpret_linear_term(l[2],v)
        return [t2-t1]
    elif l[0] == '>=':
        assert len(l) == 3
        t1 = _interpret_linear_term(l[1],v)
        t2 = _interpret_linear_term(l[2],v)
        return [t1-t2]
    elif l[0] == '=':
        assert len(l) == 3
        t1 = _interpret_linear_term(l[1],v)
        t2 = _interpret_linear_term(l[2],v)
        return [t2-t1,t1-t2]
    else:
        raise Exception('Do not understand inequality %s' % l)

# Make the pyparsing parser which tokenizes an input string into a Python list
def _make_parser():
    LP = pp.Literal("(").suppress()
    RP = pp.Literal(")").suppress()

    chars = pp.alphanums + '_<>=.+-*/' 
    String = pp.Word(chars)

    SExpr = pp.Forward()

    SExprList = pp.Group(pp.ZeroOrMore(SExpr | String))
    SExpr << (LP + SExprList + RP)

    return SExpr

############### PUBLIC INTERFACE ###############

# Parse smt2 (lisp)-style expression, returning a nested list of the tokens.
def parse(s):
    parser = _make_parser()
    res = parser.parseString(s)
    return res.asList()

# Interpret a list of parsed inequalities, returns an array of coefficients as follows:
# the output is a numpy array of shape C x (N+1), where each row r = (r_1, ..., r_{N+1}) encodes an inequality: x_1 r_1 + ... + x_N r_N + r_{N+1} >= 0.
# (C may differ from len(l) since an equality is rendered as two inequalities <=, >=).
# Inequalities are understood as conjunctions, the conjunction keyword 'and' is effectively ignored.
def interpret(l,v):
    assert type(l) is list
    ineqs = []
    for ineq in l:
        ineqs.extend(_interpret_inequality(ineq,v))

    return np.stack(ineqs,0)

# Parse and interpret a string containing inequalities in the variables v (which determine their order).
# See interpret (above) for details on the return value.
def parse_and_interpret(s,v):
    l = parse(s)
    assert len(l) == 1 and l[0][0] == 'and'
    return interpret(l,v)
