"""
Basic module to parse an smt2 file which contains a conjunction of inequalities into a numpy array. The output 
of the functions interpret or parse_and_interpret is a numpy array of shape C x (N+1), where C is the number
of inequalities, and each row r = (r_1, ..., r_{N+1}) encodes an inequality: x_1 r_1 + ... + x_N r_N + r_{N+1} >= 0.
"""

import numpy as np
import sys
import pyparsing as pp
import argparse

# Obtain all variables mentioned in the passed parsed list.
def _obtain_variables(l):
    all_vars = set()
    if type(l) is list:
        for el in l:
            all_vars.update(_obtain_variables(el))
    else:
        if type(l) is str and l.startswith('x'):
            all_vars.update([l])
    return list(all_vars)

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

# Parse and interpret a string containing inequalities, if v (the variables) are not given, they are
# detected from the string (but in random order). Returns the interpretation (see interpret function above)
# and the names of the variables, in the order in which they are encoded in the interpretation.
# If the list of variables is passed as input, this list is used in order to sort the variables.
# See interpret (above) for details on the return value.
def parse_and_interpret(s,v=None):
    l = parse(s)
    assert len(l) == 1 and l[0][0] == 'and'
    if v is None:
        v = _obtain_variables(l)
    return interpret(l,v),v

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Script to parse an smt2 file into a numpy array.')
    argparser.add_argument('-t','--test',action='store_true')
    args = argparser.parse_args()

    if not args.test:
        print('This script is not intended to be called directly, unless for testing (--test)')
        argparser.print_help()
        sys.exit()
    else:
        print('Running tests, part 1 (Python lists)')
        # linear terms
        v = ['x1','x2','x13']
        t1 = ['+',3.5,6.7]; r1=[0.0,0.0,0.0,10.2]
        np.testing.assert_almost_equal(_interpret_linear_term(t1,v),r1)

        t2 = ['+',['*',4,-3.3],6.7]; r2=[0.0,0.0,0.0,-6.5]
        np.testing.assert_almost_equal(_interpret_linear_term(t2,v),r2)

        t3 = ['-',['+',['*',['/',5,2],'x1'],['*','x13',3.3]],6.6]; r3=[2.5,0.0,3.3,-6.6]
        np.testing.assert_almost_equal(_interpret_linear_term(t3,v),r3)

        t4 = ['+','x1','x13']; r4 = [1.0,0.0,1.0,0.0]
        np.testing.assert_almost_equal(_interpret_linear_term(t4,v),r4)

        t5a = ['-', ['+',['/','x1',3],['*','x2',2.2]],['*',5.1,'x13']]
        t5b = ['+', 'x1','x13']
        t5 = ['+',['/',t5a,['*',2,3.0]],t5b]
        # (1/3 x_1 + 2.2x_2 -5.1x_3) / 6 + (x_1+x_3)
        r5 = [19/18,2.2/6,-5.1/6+1,0.0]
        np.testing.assert_almost_equal(_interpret_linear_term(t5,v),r5)

        print('Running tests, part 2 (SMT data)')
        v = ['x2','x5','x13']

        s1 = '(and (<= x2 x13))'
        r1 = [[ -1,0,1,0 ]]
        np.testing.assert_almost_equal(parse_and_interpret(s1,v)[0],r1)

        s2 = '(and (<= x2 (+ (* 2 x13) 3.4)) (not (<= (/ x5 3) x2 )) )'
        r2 = [
                [ -1,0,2,3.4 ],
                [ -1,1/3,0,0]
                ]
        np.testing.assert_almost_equal(parse_and_interpret(s2,v)[0],r2)

        # long plus
        s3 = '(and (<= 3 (+ x2 (* 2 x13) (* 9.5 x5) ) ))'
        r3 = [
                [ 1,9.5,2,-3 ],
                ]
        np.testing.assert_almost_equal(parse_and_interpret(s3,v)[0],r3)

        # different types of (in)equalities
        s4 = '(and (>= 4 (+ x5 (* -2.2 x2) ) ) (<= x2 2))'
        r4 = [
                [ 2.2,-1,0,4 ],
                [ -1,0,0,2 ]
                ]
        np.testing.assert_almost_equal(parse_and_interpret(s4,v)[0],r4)
        
        s5 = '(and (= x5 (* 2.2 x2)))'
        r5a = [ [ 2.2, -1, 0, 0], [ -2.2, 1, 0, 0] ]
        r5b = [ [ -2.2, 1, 0, 0], [ 2.2, -1, 0, 0] ]
        np.testing.assert_almost_equal(parse_and_interpret(s5,v)[0],r5a) # r5b is also ok

        # many variables
        v_long = ['x2','x5','x6','x8','x13','x14' ]
        s7 = '(and (<= x2 (+ (* 2 x13) 3.4)) (not (<= (/ x5 3) x2 )) )'
        r7 = [
                [ -1,0,0,0,2,0,3.4 ],
                [ -1,1/3,0,0,0,0,0]
                ]
        np.testing.assert_almost_equal(parse_and_interpret(s7,v_long)[0],r7)

        # spurious ands
        s8 = '(and (<= x2 2) (and (>= x5 (* 2.2 x13)) (<= x5 777.7)) (>= x13 4.4))'
        r8 = [
                [ -1,0,0,2 ],[0,1,-2.2,0],[0,-1,0,777.7],[0,0,1,-4.4]
                ]
        np.testing.assert_almost_equal(parse_and_interpret(s8,v)[0],r8)

        print('Everything OK')

