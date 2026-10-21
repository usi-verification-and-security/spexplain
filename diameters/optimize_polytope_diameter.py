import parse_smt2
import plot_polytope
import find_polytope_bounds
import scipy.optimize

import numpy as np
import sys
import argparse

import matplotlib.pyplot as plt
import matplotlib.lines as lines

# optimize diameter of a polytope in two dimensions
def optimize_diameter_2d(ineq,var_start,var_bounds):
    # function to be optimized
    def line_len_sq(x):
        # x should be a four-component vector [point1_x,point1_y,point2_x,point2_y]
        return (x[0] - x[2]) ** 2 + (x[1] - x[3]) ** 2

    # compute constraints from inequalities
    constraints = []
    for pos in range(ineq.shape[0]):
        this_constraint = ineq[pos]
        # this_constraint is a 3-component vector [a,b,c], enconding the constraint
        # ax+by+c>=0 <=> ax+by >= -c
        # note that constraints must be valid for both points
        mx = np.array([ [this_constraint[0],this_constraint[1],0,0],[0,0,this_constraint[0],this_constraint[1]] ] )
        lb = -this_constraint[2]
        ub = np.inf

        constraints.append(scipy.optimize.LinearConstraint(mx,lb,ub))

    # make bounds
    bounds = [ var_bounds[0],var_bounds[1], var_bounds[0],var_bounds[1] ] # var_bounds[x] is a pair

    # make start
    start = np.concatenate(var_start)

    # perform opt, note that we takenegative to maximize
    result = scipy.optimize.minimize(lambda x: -line_len_sq(x),start,bounds=bounds,constraints=constraints)

    return (result.x[0:2],result.x[2:4])

# optimize diameter of a polytope in N dimensions (inferred from size of input vars)
def optimize_diameter(ineq,var_start,var_bounds,extra_constraints=[]):
    # function to be optimized
    def line_len_sq(x):
        # x should be a vector of 2N components (for dimensionality N), first N components = first point, rest = second point
        mid = x.shape[0] // 2
        p1 = x[:mid]
        p2 = x[mid:]
        print('LLS: ',np.dot(p1-p2,p1-p2))
        return np.dot(p1-p2,p1-p2)
#         return np.linalg.norm(p1-p2)

    # compute constraints from inequalities
    constraints = []
    N = ineq.shape[1] - 1

    for pos in range(ineq.shape[0]):
        this_constraint = ineq[pos]
        # this_constraint is an N+1-component vector [p1,...,pN,c], enconding the constraint
        # p1x1 + p2x2 + ... + pNxN +c >= 0 <=> p1x1 + p2x2 + ... + pNxN >= -c
        # note that constraints must be valid for both points of the diameter to be optimized,
        # so we have two components in each constraint
        pad = np.zeros((N,),dtype=ineq.dtype)

        cons_p1 = np.concatenate([this_constraint[:-1],np.zeros((N,),dtype=ineq.dtype)],0)
        cons_p2 = np.concatenate([np.zeros((N,),dtype=ineq.dtype),this_constraint[:-1]],0)

        mx = np.stack([cons_p1,cons_p2],0)
        lb = -this_constraint[2]
        ub = np.inf

        constraints.append(scipy.optimize.LinearConstraint(mx,lb,ub))

    for ec in extra_constraints:
        constraints.append(scipy.optimize.LinearConstraint(ec[0],ec[1],ec[2]))

    # make bounds
    bounds = [ var_bounds[p] for p in range(len(var_bounds)) ] * 2

    # make start
    var_start = np.array(var_start)
    start_scale = np.linalg.norm(var_start) / 10
    start2 = var_start + start_scale * np.random.standard_normal(size=var_start.shape)
    start = np.concatenate([var_start,start2],0)
    print('Starting opt from',start)

    # perform opt, note that we take negative to maximize
    neg_len = lambda x: -line_len_sq(x)
    result = scipy.optimize.minimize(neg_len,start,bounds=bounds,constraints=constraints,method='SLSQP')

    if not result.success:
        raise Exception('Optimization could not be performed, message is %s' % str(result))
    assert result.x.shape[0] == 2*N
    return (result.x[:N],result.x[N:])

def plot_line(axes,p1,p2,color):
    plot_x = [p1[0],p2[0]]
    plot_y = [p1[1],p2[1]]
    ax.plot(plot_x,plot_y,color)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Optimize polytope diameter, given a polytope in an SMT2 file.')
    argparser.add_argument('-t','--test',action='store_true')
    argparser.add_argument('-p','--plot',action='store_true',help='Plot result')
    argparser.add_argument('filename',nargs='?')
    args = argparser.parse_args()

    if args.test:
        ###
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
# #         start1 = ( [2,0],[2.2,0.3] ) # two points which are within the polytope
#         start1 = [0.5,0] # point within the polytope
# 
#         parsed_ineq1 = parse_smt2.parse_and_interpret(ineq1,vars1)[0]
# 
#         # plot basis
#         ax = plot_polytope.plot_2d(parsed_ineq1,bounds1[0],bounds1[1],interactive=True)
# 
#         # perform optimization, result is a tuple (point1,point2)
#         opt_result1 = optimize_diameter(parsed_ineq1,start1,bounds1) 
# 
#         # also get best diameter orthogonal to this one
#         orth_dir = opt_result1[1] - opt_result1[0]
#         orth_cons = np.concatenate([orth_dir,-orth_dir],0)[None]
#         opt_result1o = optimize_diameter(parsed_ineq1,start1,bounds1,[(orth_cons,0.0,0.0)]) 
#         
#         # plot result
# #         plot_x = [opt_result1[0][0],opt_result1[1][0]]
# #         plot_y = [opt_result1[0][1],opt_result1[1][1]]
# #         ax.plot(plot_x,plot_y,color='red')
#         plot_line(ax,opt_result1[0],opt_result1[1],'red')
#         plot_line(ax,opt_result1o[0],opt_result1o[1],'green')
#         ax.scatter(start1[0],start1[1],c='yellow')
# 
#         plt.title('Optimizing diameter of triangle') 
# 
#         ax.set_aspect('equal')
#         plt.show()
         
        # square
        # 
        ineq2 = """
            (and 
                (<= 0 x1) (<= x1 1)
                (<= 2 x2) (<= x2 3)
            )
        """
        ineq2 = """
            (and 
                (<= -.4 x1) (<= x1 .4)
                (<= -.4 x2) (<= x2 .4)
            )
        """
        ineq2 = """
            (and 
                (<= 0 x1) (<= x1 .4)
                (<= 0 x2) (<= x2 .4)
            )
        """
        bounds2 = ( [-.5,3.5], [-.5,3.5] )
        vars2 = [ 'x1','x2' ]
#         start2 = ( [0.5,0.5],[0.51,0.51] ) # two points which are within the polytope
        start2 = ( [0.5,2.5] ) # point in the polytope
        start2 = ( [0.1,0.1] ) # point in the polytope

        parsed_ineq2 = parse_smt2.parse_and_interpret(ineq2,vars2)[0]

        # plot basis
        ax = plot_polytope.plot_2d(parsed_ineq2,bounds2[0],bounds2[1],interactive=True)

        # perform optimization, result is a tuple (point1,point2)
        opt_result2 = optimize_diameter(parsed_ineq2,start2,bounds2) 
        
        # also get best diameter orthogonal to this one
        orth_dir = opt_result2[1] - opt_result2[0]
        orth_cons = np.concatenate([orth_dir,-orth_dir],0)[None]
        opt_result2o = optimize_diameter(parsed_ineq2,start2,bounds2,[(orth_cons,0.0,0.0)]) 

        # plot result
        plot_line(ax,opt_result2[0],opt_result2[1],'red')
        plot_line(ax,opt_result2o[0],opt_result2o[1],'green')
        ax.scatter(start2[0],start2[1],c='yellow')

        plt.title('Optimizing diameter of square') 

        ax.set_aspect('equal')
        plt.show()
        # pentagon:
        # 
        ineq3 = """
            (and 
                (<= (* 0.1 (+ x1 1)) x2)
                (<= (- x1 .3) x2 )
                (<= (+ (* 5  (* -1 x1)) 1) x2 )
                (<= x2 (+ (* 1.6 x1) .4)  )
                (<= x2 (+ (* -0.5 x1) .8)  )
            )
        """
        bounds3 = ( [0,1], [0,1] )
        vars3 = [ 'x1','x2' ]
#         start2 = ( [0.5,0.5],[0.51,0.51] ) # two points which are within the polytope
        start3 = ( [0.5,0.5] ) # two points which are within the polytope

        parsed_ineq3 = parse_smt2.parse_and_interpret(ineq3,vars3)[0]

        # plot basis
        ax = plot_polytope.plot_2d(parsed_ineq3,bounds3[0],bounds3[1],interactive=True)

        # perform optimization, result is a tuple (point1,point2)
        opt_result3 = optimize_diameter(parsed_ineq3,start3,bounds3) 
        
        # also get best diameter orthogonal to this one
        orth_dir = opt_result3[1] - opt_result3[0]
        orth_cons = np.concatenate([orth_dir,-orth_dir],0)[None]
        opt_result3o = optimize_diameter(parsed_ineq3,start3,bounds3,[(orth_cons,0.0,0.0)]) 

        # plot result
        plot_line(ax,opt_result3[0],opt_result3[1],'red')
        plot_line(ax,opt_result3o[0],opt_result3o[1],'green')
        ax.scatter(start3[0],start3[1],c='yellow')

        plt.title('Optimizing diameter of pentagon') 

        ax.set_aspect('equal')
        plt.show()
    else:
        # needs filename
        if args.filename is None :
            print('If --test is not given, a filename must be given')
            argparser.print_help()
            sys.exit()
        data = open(args.filename,'rt').read()
        data = ''.join(data.splitlines())

        parsed_data,parsed_vars = parse_smt2.parse_and_interpret(data,None)
         
        # find bounds and starting point
        bounds = find_polytope_bounds.find_bounds(parsed_data)
        start = find_polytope_bounds.find_valid_points(parsed_data)

        opt_result = optimize_diameter_2d(parsed_data,start,bounds) # TODO

        print('Found diameter from (%f,%f) to (%f,%f)' % (opt_result[0][0],opt_result[0][1],opt_result[1][0],opt_result[1][1]))
        dlt = opt_result[1] - opt_result[0]
        print('Length: %f' % np.sqrt(np.dot(dlt,dlt)))

        if args.plot:
            plot_bounds = bounds + np.array([-1,1])[None,:]

            ax = plot_polytope.plot_2d(parsed_data,bounds[0],bounds[1],interactive=True)

            # plot result
            plot_x = [opt_result[0][0],opt_result[1][0]]
            plot_y = [opt_result[0][1],opt_result[1][1]]
            ax.plot(plot_x,plot_y,color='red')
            ax.set_aspect('equal')

            plt.title('Optimizing diameter (2D)')

            plt.show()
