import numpy as np
import sys
import pyparsing as pp

# obtain all variables mentioned in the passed parsed list
def _obtain_variables(l):
    all_vars = set()
    if type(l) is list:
        for el in l:
            all_vars.update(_obtain_variables(el))
    else:
        if type(l) is str and l.startswith('x'):
            all_vars.update([l])
    return list(all_vars)

# recursively interpret a list l of parsed linear terms in variables v, returns an array of shape (len(v) + 1), where
# the last element is the constant term, and the array encodes the linear combination in l, i.e.
# r = _interpret_linear_term(l,v), then l is equivalent to sum_i r_i v_i, where the last element of v is the constant
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
            val = int(l)
            res = np.zeros(len(v) + 1)
            res[-1] = val
            return res
    else:
        assert type(l) is list, '%s is not a list' % str(l)
        term1 = _interpret_linear_term(l[1],v)

        if l[0] == '+':
            for term in l[2:]:
                term1 += _interpret_linear_term(term,v)
            return term1

        assert len(l) == 2 or len(l) == 3
        if len(l) == 2:
            term2 = None #special case: unary -
        else:
            term2 = _interpret_linear_term(l[2],v)
        # if l[0] == '+':
        #     return term1 + term2
        if l[0] == '-':
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

# interpret a parsed inequality of the form ('<=',t1,t2), returns an array of shape (len(v) + 1), where
# the last element is the constant term, and the array encodes the inequality in l, i.e.
# r = _interpret_inequality(l,v), then r = t2-t1, and the inequality is r >= 0
def _interpret_inequality(l,v):
    # print('Interpreting inequality',l)
    assert len(l) == 2 or len(l) == 3
    if len(l) == 2:
        assert l[0] == 'not'
        return -_interpret_inequality(l[1],v)
    else:
        if l[0] == '>=':
            l[0] = '<='
            l[1],l[2] = l[2],l[1]

        assert len(l) == 3 and l[0] in  ['<=', '=']
        t1 = _interpret_linear_term(l[1],v)
        t2 = _interpret_linear_term(l[2],v)
        return t1 - t2

def _make_parser():
    LP = pp.Literal("(").suppress()
    RP = pp.Literal(")").suppress()

    chars = pp.alphanums + '_<>=.+-*/'
    String = pp.Word(chars)
    #     SingleQuoteString = pp.QuotedString(quoteChar="'", unquoteResults=False)
    #     DoubleQuoteString = pp.QuotedString(quoteChar='"', unquoteResults=False)
    #     QuotedString = SingleQuoteString | DoubleQuoteString
    #     Atom = String | QuotedString
    SExpr = pp.Forward()
    #     SExprList = pp.Group(pp.ZeroOrMore(SExpr | Atom))
    SExprList = pp.Group(pp.ZeroOrMore(SExpr | String))
    SExpr << (LP + SExprList + RP)

    return SExpr

# parse lisp-style expression
def parse(s):
    parser = _make_parser()
    res = parser.parseString(s)
    return res.asList()

# interpret a list of parsed inequalities, returns an array of coefficients...
def interpret(l,v):
    assert l[0] in {'and', 'not'}

    if l[0] == 'not':
        assert len(l) == 2
        return interpret(['and', l],v)

    ineqs = []
    eqs = []
    for i in l[1:]:
        if i[0] == 'and':
            assert l[0] != 'not'
            ineqs1, eqs1 = interpret(i,v)
            ineqs.extend(ineqs1)
            eqs.extend(eqs1)
        elif i[0] == '=':
            eqs.append(_interpret_inequality(i,v))
        else:
            ineqs.append(_interpret_inequality(i,v))
    # ineqs = [ _interpret_inequality(i,v) for i in l[1:] ]
    if ineqs:
        ineqs_array = np.stack(ineqs, 0)
    else:
        ineqs_array = np.empty((0, len(v) + 1))

    if eqs:
        eqs_array = np.stack(eqs, 0)
    else:
        eqs_array = np.empty((0, len(v) + 1))
    return ineqs_array, eqs_array


# parse a string containing inequalities, if v (the variables) are not given, they are
# detected from the string (but in random order)
def parse_and_interpret(s,v=None):
    l = parse(s)
    assert len(l) == 1
    l = l[0]
    if v is None:
        v = _obtain_variables(l)

    if l[0] == 'or':
        conjucts = [interpret(i,v) for i in l[1:]]
        return conjucts, v

    ineq, eq = interpret(l,v)
    return [(ineq, eq)], v


if __name__ == '__main__':
    if sys.argv[1] == '--test':
        # linear terms
        v = ['x1','x2','x3']
        t1 = ['+',3.5,6.7]; r1=[0.0,0.0,0.0,10.2]
        np.testing.assert_almost_equal(_interpret_linear_term(t1,v),r1)

        t2 = ['+',['*',4,-3.3],6.7]; r2=[0.0,0.0,0.0,-6.5]
        np.testing.assert_almost_equal(_interpret_linear_term(t2,v),r2)

        t3 = ['-',['+',['*',['/',5,2],'x1'],['*','x3',3.3]],6.6]; r3=[2.5,0.0,3.3,-6.6]
        np.testing.assert_almost_equal(_interpret_linear_term(t3,v),r3)

        t4 = ['+','x1','x3']; r4 = [1.0,0.0,1.0,0.0]
        np.testing.assert_almost_equal(_interpret_linear_term(t4,v),r4)

        t5a = ['-', ['+',['/','x1',3],['*','x2',2.2]],['*',5.1,'x3']]
        t5b = ['+', 'x1','x3']
        t5 = ['+',['/',t5a,['*',2,3.0]],t5b]
        # (1/3 x_1 + 2.2x_2 -5.1x_3) / 6 + (x_1+x_3)
        r5 = [19/18,2.2/6,-5.1/6+1,0.0]
        np.testing.assert_almost_equal(_interpret_linear_term(t5,v),r5)
    else:
        data = open(sys.argv[1],'rt').read()
        data = ''.join(data.splitlines())
        parse_and_interpret(data)