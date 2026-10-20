"""
Methods to understand the shape of a polytope. Currently can determine valid points
(e.g. for further optimization), as well as bounds for the entire polytope.
"""
import parse_smt2
import plot_polytope
import scipy.optimize
import matplotlib.pyplot as plt

import numpy as np
import sys
import argparse

############### PUBLIC INTERFACE ###############

# Find two N-dimensional valid points, i.e. points which are within the 
# area defined by the conjunction of the inequalities. Points might be
# near the boundaries of the region. ineq must be given as Numpy array.
# Throws exception if something goes amiss, otherwise returns a 2-tuple of points.
def find_valid_points(ineq):
    # encode as linear program, minimize \sum_i x_i and \sum_i -xi
    # in order to find two points
    N = ineq.shape[1] - 1 # last dimension is the constant

    A_ub = -ineq[:,:-1] # Ax+b > 0 <=> Ax > -b <=> -Ax < b
    b_ub = ineq[:,-1]

    # one point
    bounds = (None,None)
    c1 = np.full((N,),1.0,dtype=ineq.dtype)
    res1 = scipy.optimize.linprog(c1,A_ub,b_ub,bounds=bounds)
    if res1.status != 0:
        raise Exception('Error finding valid point: %s' % str(res1))
    point1 = res1.x

    c2 = np.full((N,),-1.0,dtype=ineq.dtype)
    res2 = scipy.optimize.linprog(c2,A_ub,b_ub,bounds=bounds)
    if res2.status != 0:
        raise Exception('Error finding valid point: %s' % str(res2))
    point2 = res2.x

    return (point1,point2)

# TODO add doc
def find_inner_point(ineq,tol=0.01):
    # first find extremal points
    extremal_points = []

    # encode as linear program, minimize \sum_i x_i and \sum_i -xi
    # in order to find two points
    N = ineq.shape[1] - 1 # last dimension is the constant

    A_ub = -ineq[:,:-1] # Ax+b > 0 <=> Ax > -b <=> -Ax < b
    b_ub = ineq[:,-1]

    # iterate over dimensions
    for dim in range(N):
        bounds = (None,None)
        c = np.zeros((N,),dtype=ineq.dtype)

        # lower bound, remember that we MINIMIZE
        c[dim] = 1.0
        lower_bound_res = scipy.optimize.linprog(c,A_ub,b_ub,bounds=bounds)
        if lower_bound_res.status == 2:
            raise Exception('Problem appears to be infeasible (no points in polytope?): %s' % str(lower_bound_res))
        elif lower_bound_res.status == 3:
            raise Exception('not yet implemented')
        else:
            extremal_points.append(lower_bound_res.x)

        # upper bound, remember that we MINIMIZE
        c[dim] = -1.0
        upper_bound_res = scipy.optimize.linprog(c,A_ub,b_ub,bounds=bounds)
        if upper_bound_res.status == 2:
            raise Exception('Problem appears to be infeasible (no points in polytope?): %s' % str(upper_bound_res))
        elif upper_bound_res.status == 3:
            raise Exception('not yet implemented')
        else:
            extremal_points.append(upper_bound_res.x)

    # simply take the average...
    candidate = np.mean(extremal_points,0)

    # test whether this point is not on a boundary
    eval_ineq = np.dot(ineq,np.concatenate([candidate,[1.0]],0))
    if any(eval_ineq < tol):
        raise Exception('could not find inner point')
    else:
        return candidate

# Find the bounds of the polytope. Returns an array of shape N*2, where N is the dimensionality
# and each row contains upper and lower bounds. Might be np.inf or -np.inf if unbounded.
def find_bounds(ineq):
    # encode as linear program, minimize \sum_i x_i and \sum_i -xi
    # in order to find two points
    N = ineq.shape[1] - 1 # last dimension is the constant

    A_ub = -ineq[:,:-1] # Ax+b > 0 <=> Ax > -b <=> -Ax < b
    b_ub = ineq[:,-1]

    result = np.zeros((N,2),dtype=ineq.dtype)

    # iterate over dimensions
    for dim in range(N):
        bounds = (None,None)
        c = np.zeros((N,),dtype=ineq.dtype)

        # lower bound, remember that we MINIMIZE
        c[dim] = 1.0
        lower_bound_res = scipy.optimize.linprog(c,A_ub,b_ub,bounds=bounds)
        if lower_bound_res.status == 2:
            raise Exception('Problem appears to be infeasible (no points in polytope?): %s' % str(lower_bound_res))
        elif lower_bound_res.status == 3:
            lower_bound = -np.inf
        else:
            lower_bound = lower_bound_res.x[dim]

        # upper bound, remember that we MINIMIZE
        c[dim] = -1.0
        upper_bound_res = scipy.optimize.linprog(c,A_ub,b_ub,bounds=bounds)
        if upper_bound_res.status == 2:
            raise Exception('Problem appears to be infeasible (no points in polytope?): %s' % str(upper_bound_res))
        elif upper_bound_res.status == 3:
            upper_bound = np.inf
        else:
            upper_bound = upper_bound_res.x[dim]

        result[dim,0] = lower_bound
        result[dim,1] = upper_bound

    return result

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Find bounds of a polytope, and/or find valid points within a polytope.')
    argparser.add_argument('-t','--test',action='store_true',help='Run tests, ignore filename parameter')
    argparser.add_argument('-p','--points',action='store_true',help='Find valid points in polytope')
    argparser.add_argument('-b','--bounds',action='store_true',help='Find bounds of polytope')
    argparser.add_argument('-i','--inner',action='store_true',help='Find an inner point of polytope')
    argparser.add_argument('filename',nargs='?')
    args = argparser.parse_args()
    if int(args.points) + int(args.bounds) + int(args.inner) + int(args.test) != 1:
        print('Exactly one of --test, --points, --inner, or --bounds must be given.')
        argparser.print_help()
        sys.exit()

    if (args.points or args.bounds or args.inner) and (args.filename is None):
        print('If --test is not given, filename argument must be given.')
        argparser.print_help()
        sys.exit()

    # Now do something useful
    if args.points:
        data = open(args.filename,'rt').read()
        data = ''.join(data.splitlines())

        parsed_data,parsed_vars = parse_smt2.parse_and_interpret(data,None)

        points = find_valid_points(parsed_data)

        print('Point 1: %s = %s' % (', '.join(parsed_vars),points[0].tolist()))
        print('Point 2: %s = %s' % (', '.join(parsed_vars),points[1].tolist()))
        sys.exit()

    elif args.bounds:
        data = open(args.filename,'rt').read()
        data = ''.join(data.splitlines())

        parsed_data,parsed_vars = parse_smt2.parse_and_interpret(data,None)

        bounds = find_bounds(parsed_data)

        print('Variable bounds:')
        for pos in range(len(parsed_vars)):
            print('%s = %f ... %f' % (parsed_vars[pos],bounds[pos][0],bounds[pos][1]))

        sys.exit()

    elif args.inner:
        data = open(args.filename,'rt').read()
        data = ''.join(data.splitlines())

        parsed_data,parsed_vars = parse_smt2.parse_and_interpret(data,None)

        point = find_inner_point(parsed_data)

        print('Inner point: ',point)

        sys.exit()


    elif args.test:
        print('Running tests for finding inner point')
        # triangle
        ineq1 = """
            (and 
                (<= x5 x4 )
                (<= x4 (+ (/ 6 2) (* 0.1 x5 ) ) )
                (<= (- x5) (* 2 x4)) 
            )
        """
        bounds1 = ( [-4,4], [-7,5] )
        vars1 = [ 'x4','x5' ]
        parsed_ineq1 = parse_smt2.parse_and_interpret(ineq1,vars1)[0]

        point = find_inner_point(parsed_ineq1)

        ax = plot_polytope.plot_2d(parsed_ineq1,bounds1[0],bounds1[1],interactive=True)

        # plot the two points
        ax.scatter(point[0],point[1],color='red')

        plt.title('Inner point - triangle') 

        plt.show()

        # pentagon
        ineq2 = """
            (and 
                (<= (* 0.1 (+ x1 1)) x2)
                (<= (- x1 .3) x2 )
                (<= (+ (* 5  (* -1 x1)) 1) x2 )
                (<= x2 (+ (* 1.6 x1) .4)  )
                (<= x2 (+ (* -0.5 x1) .8)  )
            )
        """
        bounds2 = ( [0,1], [0,1] )
        vars2 = [ 'x1','x2' ]

        parsed_ineq2 = parse_smt2.parse_and_interpret(ineq2,vars2)[0]

        point = find_inner_point(parsed_ineq2)

        ax = plot_polytope.plot_2d(parsed_ineq2,bounds2[0],bounds2[1],interactive=True)

        # plot the two points
        ax.scatter(point[0],point[1],color='red')

        plt.title('Inner point - pentagon') 

        plt.show()

        ineq3 = """
            (and
                (<= 0 x3)
                (<= (- x3 4) (+ (* 2.0 x1) (* 2.0 x2)))
                (<= (- x3 4) (+ (* -2.0 x1) (* 2.0 x2)))
                (<= (- x3 4) (+ (* 2.0 x1) (* -2.0 x2)))
                (<= (- x3 4) (+ (* -2.0 x1) (* -2.0 x2)))
                )
                """

        vars3 = ['x1','x2','x3' ]

        parsed_ineq3 = parse_smt2.parse_and_interpret(ineq3,vars3)[0]

        point = find_inner_point(parsed_ineq3)

        print('Inner point in pyramid:',point)

#         print('Running tests for finding valid points')
#         ###
#         # triangle
#         ineq1 = """
#             (and 
#                 (<= x5 x4 )
#                 (<= x4 (+ (/ 6 2) (* 0.1 x5 ) ) )
#                 (<= (- x5) (* 2 x4)) 
#             )
#         """
#         bounds1 = ( [-4,4], [-7,5] )
#         vars1 = [ 'x4','x5' ]
#         parsed_ineq1 = parse_smt2.parse_and_interpret(ineq1,vars1)[0]
# 
#         points = find_valid_points(parsed_ineq1)
# 
#         ax = plot_polytope.plot_2d(parsed_ineq1,bounds1[0],bounds1[1],interactive=True)
# 
#         # plot the two points
#         ax.scatter(points[0][0],points[0][1],color='red')
#         ax.scatter(points[1][0],points[1][1],color='orange')
# 
#         plt.title('Valid points - triangle') 
# 
#         plt.show()
#         
#         # pentagon
#         ineq2 = """
#             (and 
#                 (<= (* 0.1 (+ x1 1)) x2)
#                 (<= (- x1 .3) x2 )
#                 (<= (+ (* 5  (* -1 x1)) 1) x2 )
#                 (<= x2 (+ (* 1.6 x1) .4)  )
#                 (<= x2 (+ (* -0.5 x1) .8)  )
#             )
#         """
#         vars2 = [ 'x1','x2' ]
#         bounds2 = ( [0,1], [0,1] )
# 
#         parsed_ineq2 = parse_smt2.parse_and_interpret(ineq2,vars2)[0]
# 
#         points = find_valid_points(parsed_ineq2)
# 
#         ax = plot_polytope.plot_2d(parsed_ineq2,bounds2[0],bounds2[1],interactive=True)
# 
#         # plot the two points
#         ax.scatter(points[0][0],points[0][1],color='red')
#         ax.scatter(points[1][0],points[1][1],color='orange')
# 
#         plt.title('Valid points - pentagon') 
# 
#         plt.show()
# 
#         print('Running tests for finding bounds')
#         ###
#       # triangle
#         ineq1 = """
#             (and 
#                 (<= x5 x4 )
#                 (<= x4 (/ 6 2) )
#                 (<= (- x5) (* 2 x4)) 
#             )
#         """
#         vars1 = [ 'x4','x5' ]
#         parsed_ineq1 = parse_smt2.parse_and_interpret(ineq1,vars1)[0]
# 
#         bounds1 = find_bounds(parsed_ineq1)
# 
#         # now let's try to plot!
#         plot_bounds1 = bounds1 + np.array([-1,1])[None,:]
#         ax = plot_polytope.plot_2d(parsed_ineq1,plot_bounds1[0],plot_bounds1[1],interactive=True)
# 
#         # plot the bounds as lines in red
#         low_x = [bounds1[0][0],bounds1[0][1]]
#         low_y = [bounds1[1][0],bounds1[1][0]]
#         ax.plot(low_x,low_y,color='red')
# 
#         up_x = [bounds1[0][0],bounds1[0][1]]
#         up_y = [bounds1[1][1],bounds1[1][1]]
#         ax.plot(up_x,up_y,color='red')
# 
#         left_x = [bounds1[0][0],bounds1[0][0]]
#         left_y = [bounds1[1][0],bounds1[1][1]]
#         ax.plot(left_x,left_y,color='red')
# 
#         right_x = [bounds1[0][1],bounds1[0][1]]
#         right_y = [bounds1[1][0],bounds1[1][1]]
#         ax.plot(right_x,right_y,color='red')
# 
#         plt.title('Polytope bounds') 
# 
#         plt.show()
# 
#         # part of upper left quadrangle, not finite
#         ineq1 = """
#             (and 
#                 (<= x1 -1)
#                 (<= 1 x2)
#                 (<= x2 3))
#             )
#         """
#         vars1 = [ 'x1','x2' ]
#         parsed_ineq1 = parse_smt2.parse_and_interpret(ineq1,vars1)[0]
# 
#         bounds1 = find_bounds(parsed_ineq1)
# 
#         # now let's try to plot!
#         plot_bounds1 = np.array([[-4,4],[-4,4]])
#         ax = plot_polytope.plot_2d(parsed_ineq1,plot_bounds1[0],plot_bounds1[1],interactive=True)
# 
#         # plot the bounds as lines in red, getting rid of infinities
#         bounds1 = np.fmax(bounds1,-10000.0)
#         bounds1 = np.fmin(bounds1,10000.0)
#         low_x = [bounds1[0][0],bounds1[0][1]]
#         low_y = [bounds1[1][0],bounds1[1][0]]
#         ax.plot(low_x,low_y,color='red')
# 
#         up_x = [bounds1[0][0],bounds1[0][1]]
#         up_y = [bounds1[1][1],bounds1[1][1]]
#         ax.plot(up_x,up_y,color='red')
# 
#         left_x = [bounds1[0][0],bounds1[0][0]]
#         left_y = [bounds1[1][0],bounds1[1][1]]
#         ax.plot(left_x,left_y,color='red')
# 
#         right_x = [bounds1[0][1],bounds1[0][1]]
#         right_y = [bounds1[1][0],bounds1[1][1]]
#         ax.plot(right_x,right_y,color='red')
# 
#         plt.title('Polytope bounds') 
# 
#         plt.show()
# 
#         # pentagon
#         ineq2 = """
#             (and 
#                 (<= (* 0.1 (+ x1 1)) x2)
#                 (<= (- x1 .3) x2 )
#                 (<= (+ (* 5  (* -1 x1)) 1) x2 )
#                 (<= x2 (+ (* 1.6 x1) .4)  )
#                 (<= x2 (+ (* -0.5 x1) .8)  )
#             )
#         """
#         vars2 = [ 'x1','x2' ]
# 
#         parsed_ineq2 = parse_smt2.parse_and_interpret(ineq2,vars2)[0]
# 
#         bounds2 = find_bounds(parsed_ineq2)
# 
#         # now let's try to plot!
#         plot_bounds2 = bounds2 + np.array([-1,1])[None,:]
#         ax = plot_polytope.plot_2d(parsed_ineq2,plot_bounds2[0],plot_bounds2[1],interactive=True)
# 
#         # plot the bounds as lines in red
#         low_x = [bounds2[0][0],bounds2[0][1]]
#         low_y = [bounds2[1][0],bounds2[1][0]]
#         ax.plot(low_x,low_y,color='red')
# 
#         up_x = [bounds2[0][0],bounds2[0][1]]
#         up_y = [bounds2[1][1],bounds2[1][1]]
#         ax.plot(up_x,up_y,color='red')
# 
#         left_x = [bounds2[0][0],bounds2[0][0]]
#         left_y = [bounds2[1][0],bounds2[1][1]]
#         ax.plot(left_x,left_y,color='red')
# 
#         right_x = [bounds2[0][1],bounds2[0][1]]
#         right_y = [bounds2[1][0],bounds2[1][1]]
#         ax.plot(right_x,right_y,color='red')
# 
#         plt.title('Polytope bounds') 
# 
#         plt.show()
# 
#         # 3D pyramid
#         ineq3 = """
#             (and
#                 (<= 0 x3)
#                 (<= (- x3 4) (+ (* 2.0 x1) (* 2.0 x2)))
#                 (<= (- x3 4) (+ (* -2.0 x1) (* 2.0 x2)))
#                 (<= (- x3 4) (+ (* 2.0 x1) (* -2.0 x2)))
#                 (<= (- x3 4) (+ (* -2.0 x1) (* -2.0 x2)))
#                 )
#                 """
# 
#         vars3 = ['x1','x2','x3' ]
# 
#         parsed_ineq3 = parse_smt2.parse_and_interpret(ineq3,vars3)[0]
# 
#         bounds3 = find_bounds(parsed_ineq3)
# 
#         print('Pyramid bounds: ',bounds3)
#         # let us plot for x2 = 0
# 
