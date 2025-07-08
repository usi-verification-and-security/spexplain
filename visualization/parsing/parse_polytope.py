# # parse polytope, given by a set of inequalities, into numpy array
# import numpy as np
# import pyparsing as pp
# import sys
#
# def _make_parser():
#     LP = pp.Literal("(").suppress()
#     RP = pp.Literal(")").suppress()
#
#     chars = pp.alphanums + '_<>=.+-*/'
#     String = pp.Word(chars)
#     #     SingleQuoteString = pp.QuotedString(quoteChar="'", unquoteResults=False)
#     #     DoubleQuoteString = pp.QuotedString(quoteChar='"', unquoteResults=False)
#     #     QuotedString = SingleQuoteString | DoubleQuoteString
#     #     Atom = String | QuotedString
#     SExpr = pp.Forward()
#     #     SExprList = pp.Group(pp.ZeroOrMore(SExpr | Atom))
#     SExprList = pp.Group(pp.ZeroOrMore(SExpr | String))
#     SExpr << (LP + SExprList + RP)
#
#     return SExpr
#
# # perform actual parsing
# def _parse(s):
#     parser = _make_parser()
#     print('Parsing',s)
#     res = parser.parseString(s)
#     return res
#
# # compute expressions with constants, input is parsed list
# def _compute_expressions(l):
#     # recurse first, processing sublists
#     recursed = [ _compute_expressions(x) if type(x) is list else x for x in l ]
#
#     # perform arithmetics
#     if len(recursed) == 3:
#         try:
#             first_par = float(recursed[1])
#             second_par = float(recursed[2])
#             op = recursed[0]
#
#             if op == '+':
#                 res = first_par + second_par
#             elif op == '-':
#                 res = first_par - second_par
#             elif op == '*':
#                 res = first_par * second_par
#             elif op == '/':
#                 res = first_par / second_par
#             else:
#                 raise ValueError('wrong operand')
#
#             return res
#         except (ValueError,TypeError):
#             pass # did not work...
#
#     elif len(recursed) == 2:
#         try:
#             par = float(recursed[1])
#             op = recursed[0]
#             if op == '-':
#                 res = -par
#             else:
#                 raise ValueError('wrong operand')
#
#             return res
#         except (ValueError,TypeError):
#             pass # did not work...
#
#     return recursed
#
# # obtain all variables mentioned in the passed parsed list
# def _obtain_variables(l):
#     all_vars = set()
#     if type(l) is list:
#         for el in l:
#             all_vars.update(_obtain_variables(el))
#     else:
#         if type(l) is str and l.startswith('x'):
#             all_vars.update([l])
#     return list(all_vars)
#
# # parse a single summand, can be a string 'x1'
# # or a list ['*', 0.123435, 'x1']
# # returns variable name and coefficient
# def parse_single_summand(l):
#     if type(l) is string:
#         return (l,1.0)
#     elif type(l) is list:
#         assert l[0] == '*'
#         return (l[2],l[1])
#
# # parse a single linear expression, e.g. ['+', 'x1', ['*', 0.02353626626018148, 'x3']]
# # returns a vector c of coefficients, such that the expression is equivalent to c_1 v_1 + ... + c_N v_N
# # where the v_N are the variables given in v
# def _parse_linear_expression(l,v):
#     assert len(l) == 3
#
#
#
#
#
# # parse a single inequality (only "<="), e.g. ['<=',63.07060879878054, ['+', 'x1', ['*', 0.02353626626018148, 'x3']]]
# # returns a numpy array of size len(vars) + 1, where the first column is the independent coefficient, and the
# # other columns are the coefficients of the variables in order
# def _parse_inequality(l):
#     # expect: '<=', constant, var expression
#     pass
#
# # convert string to array of inequalities
# def convert(s):
#     # parse to Python list
#     s1 = _parse(s).asList()
#     print('Parsing result:',s1)
#     s2 = _compute_expressions(s1)
#     print('Computed Experssions',s2)
#     all_vars = list(set(_obtain_variables(s2)))
#     print('Variables:',all_vars)
#
# if __name__ == '__main__':
#     #     parse(" ('a' 'b' (3 4 'c'))")
#     data = open(sys.argv[1],'rt').read()
#     data = ''.join(data.splitlines())
#
#     data = "(and (<= 63.0 x1) (<= x1 63.0) (<= (- (/ 133461.0 2440.0)) (+ (* (- 1.0) x1) (* (/ 6753.0 2440.0) x3))))"
#     data = "(4 (1 63.0 2))"
#     arr = convert(data)
#
# #     # only test
# #     ineq = ['<=', 63.07060879878054, ['+', 'x1', ['*', 0.02353626626018148, 'x3']]]
# #     linexp = [ ['+', 'x1', ['*', 0.02353626626018148, 'x3']] ]
# #     variables = [ 'x1','x3' ]
# #     res = _parse_linear_expression(linexp,variables)
# #     print(res)
# #
#
