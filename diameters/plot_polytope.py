"""
Basic module to plot a polytope, usually given as SMT2 file. Can be called as a standalone script,
or be used as library. The relevant functions are 
- parse_and_plot: Plot a 2D slice of the polytope by fixing all but two variables.
- parse_and_plot_slice: Plot a 2D slice of the polytope, where the slice is given as
  plane in explicit representation (i.e. support vector and two directions).
If the "interactive" parameter is given (only in library use), the figure is not drawn to the screen.
In this case the axes are returned for further use.
"""

import numpy as np
import sklearn.decomposition
import matplotlib.pyplot as plt
import sys
import argparse

import parse_smt2

############### PUBLIC INTERFACE ###############

# Plot a polytope, given by a set of inequalities in N variables. 
# Each row r = (r_1, ..., r_{N+1}) encodes an inequality: x_1 r_1 + ... + x_N r_N + r_{N+1} >= 0.
# The plot is a 2D plot over the first two variables, the values of the other variables remain fixed
# and must be given in fixed_vars.
def plot(ineq,fixed_vars,x_bounds,y_bounds,*,axes=None,interactive=False,axes_labels=(None,None)):
    assert ineq.shape[1] - 3 == fixed_vars.shape[0] # give values for all but two variables, not for the constant

    fixed_part = np.dot(ineq[:,2:-1],fixed_vars)[:,None]

    reduced_ineq = np.concatenate([ineq[:,:2],fixed_part + ineq[:,-1:]],1)
    return plot_2d(reduced_ineq,x_bounds,y_bounds,axes=axes,interactive=interactive,axes_labels=axes_labels)

# Plot a slice of a multidimensional polytope...
def plot_slice(ineq,slice_basis,slice_dirs,x_bounds,y_bounds,*,axes=None,interactive=False,axes_labels=(None,None)):
    # project inequalities to each direction, then call plot_2d in new coordinate system
    slice_dirs_norm = slice_dirs / np.linalg.norm(slice_dirs,ord=2,axis=1,keepdims=True)

#     # TODO basis != 0
#     projected_var_part = ineq[:,:-1] @ slice_dirs_norm.T
#     projected_ineq = np.concat([projected_var_part,ineq[:,-1:]],1)

    # TODO this is really ugly FIXME
    projected_ineq = np.full((ineq.shape[0],3),np.nan,dtype=ineq.dtype)
    for pos in range(ineq.shape[0]):
        this_ineq = ineq[pos]
        proj0 = np.dot(this_ineq[:-1],slice_dirs_norm[0])
        proj1 = np.dot(this_ineq[:-1],slice_dirs_norm[1])
        projc = np.dot(this_ineq[:-1],slice_basis) + this_ineq[-1]

        projected_ineq[pos] = np.array([proj0,proj1,projc])

    return plot_2d(projected_ineq,x_bounds,y_bounds,axes=axes,interactive=interactive,axes_labels=axes_labels)

# Plot a polytope, given by a set of inequalities in two variables
# Each row of ineq encodes one inequality; it has length 3, and a row (a, b, c) encodes
# the inequality ax + by + c >= 0).
# need to pass in bounds
def plot_2d(ineq,x_bounds,y_bounds,*,axes=None,interactive=False,axes_labels=(None,None)):
    # sample 100*100 points
    x = np.linspace(x_bounds[0],x_bounds[1],num=1000)
    y = np.linspace(y_bounds[0],y_bounds[1],num=1000)
    grid = np.meshgrid(x,y)
    const = np.ones_like(grid[0])
    pts = np.stack([grid[0],grid[1],const],axis=-1).reshape((-1,3))

    # check whether these points satisfy the inequality
    ev = pts @ ineq.T
    sat = (ev >= -0.1) # which inequalities are satisified, for each point
    all_sat = np.all(sat,1)
    valid_pts = pts[all_sat][:,:2]

    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot()

    axes.scatter(valid_pts[:,0],valid_pts[:,1],s=1)
    axes.set_xlim(x_bounds[0],x_bounds[1])
    axes.set_ylim(y_bounds[0],y_bounds[1])
    if axes_labels[0] is not None:
        axes.set_xlabel(axes_labels[0])
    if axes_labels[1] is not None:
        axes.set_ylabel(axes_labels[1])

    if interactive:
        return axes
    else:
        plt.show()


# Plot a polytope given as a LISP-style expression
# Need to pass in order of variables (for correctly interpreting ineq) and the variable ranges, as dictionary e.g.:
# { 'x1': [ -1.0,1.0 ], 'x2': 3.4, 'x3': [0.0,5.0] ...}
# Exactly two ranges should be actual ranges, the others should be fixed. The order of variables in all_vars
# determines which variable is plotted on the x/y axis, it is suggested to place the
# two variables which are plotted first in the list (example above: [ 'x3', 'x1', 'x2' ...] or [ 'x1', 'x3', 'x2... ]
def parse_and_plot(ineq,all_vars,ranges,*,axes=None,interactive=False):
    # parse smt2 string
    parsed_ineq,parsed_vars = parse_smt2.parse_and_interpret(ineq,all_vars)

    # understand which variables are to be plotted, and which ones are fixed
    plotted_vars = []
    fixed_vars = []

    for var in all_vars:
        try:
            val = float(ranges[var])
            fixed_vars.append(var)
        except TypeError:
            # this is an actual range
            plotted_vars.append(var)

    assert len(plotted_vars) == 2

    # make bounds, and make the array of fixed values
    x_bounds = ranges[plotted_vars[0]]
    y_bounds = ranges[plotted_vars[1]]

    fixed_values = [ ranges[f] for f in fixed_vars ]

    # reorder ineq
    new_variable_order = [ all_vars.index(p) for p in plotted_vars ] + [ all_vars.index(f) for f in fixed_vars ] + [ -1 ]
    reordered_ineq = parsed_ineq[:,new_variable_order]

    return plot(reordered_ineq,np.array(fixed_values),x_bounds,y_bounds,axes=axes,interactive=interactive,axes_labels=(plotted_vars[0],plotted_vars[1]))
    
# def parse_and_plot_slice(ineq,all_vars,ranges,slice_params,*,axes=None,interactive=False):


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Plot a polytope, given as set of inequalities or smt2 file.')
    argparser.add_argument('-t','--test',action='store_true')
    argparser.add_argument('filename',nargs='?')
#     argparser.add_argument('bounds1',nargs='?',help='Variable and bounds of horizontal axis, in form "x2=3.4,5.6"')
#     argparser.add_argument('bounds2',nargs='?',help='Variable and bounds of vertical axis, in form "x5=-0.1,0.1"')
    argparser.add_argument('slices',nargs='*',help='Values of extra variables (to plot a slice), in form "x4=1.234 x11=5.678"')
    args = argparser.parse_args()
    if not args.test:
        # needs filename
        if args.filename is None :
            print('If --test is not given, the positional arguments (filename and variable bounds) must be given')
            argparser.print_help()
            sys.exit()
        data = open(args.filename,'rt').read()
        data = ''.join(data.splitlines())

#         # each of these should be "x4=-3.14,2.75"
#         var1,bounds1 = args.bounds1.split('=')
#         bounds1 = [ float(b) for b in bounds1.split(',') ]
#         var2,bounds2 = args.bounds2.split('=')
#         bounds2 = [ float(b) for b in bounds2.split(',') ]

        slices = {}
        variables = []
        for pos,sl in enumerate(args.slices):
            var,bound = sl.split('=')
            if pos in (0,1): # first two should be ranges
                bound = [ float(b) for b in bound.split(',') ]
            else:
                bound = float(bound)
            slices[var] = bound
            variables.append(var)

#         slices[var1] = bounds1
#         slices[var2] = bounds2

#         if len(sys.argv) == 5 and sys.argv[4] == '--animate':
#             animate = True
#         else:
#             animate = False
        animate = False # TODO

        parse_and_plot(data,variables,slices)

    else:
        # plot inequality encoded as array
    #     # test triangle: x > y, x < 3, 2x > -y
    #     ineq1 = np.array([ [ 1,-1,0 ],[-1,0,3],[2,1,0]],dtype=float)
    #     bounds1 = ( [-4,4], [-7,5] )
    #     plot_2d(ineq1,bounds1[0],bounds1[1],True)

    #     # test square: x > -1, x < 1, same for y
    #     ineq2 = np.array([ [ 1,0,1.1 ],[-1,0,1.1],[0,1,1],[0,-1,1]],dtype=float)
    #     bounds2 = ( [-4,4], [-5,5] )
    #     plot_2d(ineq2,bounds2[0],bounds2[1],True)

        # plot inequality encoded as LISP-style expression
        # CURRENT LIMITATIONS:
        # variables must have the form x[number]
        # you must additionally pass the variables in the parameters variables (otherwise we do not know which bound refers to which variable)
        # for >= you have to write (not (<= ...))

        # triangle, again the same as above
#         ineq6 = """
#             (and 
#                 (<= x5 x4 )
#                 (<= x4 (/ 6 2) )
#                 (<= (- x5) (* 2 x4)) 
#             )
#         """
#         bounds6 = ( [-4,4], [-7,5] )
#         vars6 = [ 'x4','x5' ]
#         parse_and_plot_2d(ineq6,vars6,bounds6[0],bounds6[1])
# 
#         ineq7 = """
#             (and 
#                 (not (<= x4 x5 ))
#                 (<= x4 (/ 6 2) )
#                 (<= (- x5) (* 2 x4)) 
#             )
#         """
#         bounds7 = ( [-4,4], [-7,5] )
#         vars7 = [ 'x4','x5' ]
#         parse_and_plot_2d(ineq7,vars7,bounds7[0],bounds7[1])

#         # 2D pyramid
#         ineq8 = """
#             (and 
#                 (<= (- x2 4) (* 2 x1))
#                 (<= (- x2 4) (* -2 x1))
#                 (<= 0 x2 )
#                 )
#         """
#         bounds8 = ( [ -3,3], [-1,5] )
#         vars8 = [ 'x1','x2' ]
#         parse_and_plot_2d(ineq8,vars8,bounds8[0],bounds8[1])

#         # 3D pyramid, plot cuts
#         ineq9 = """
#             (and
#                 (<= 0 x3)
#                 (<= (- x3 4) (+ (* 2.0 x1) (* 2.0 x2)))
#                 (<= (- x3 4) (+ (* -2.0 x1) (* 2.0 x2)))
#                 (<= (- x3 4) (+ (* 2.0 x1) (* -2.0 x2)))
#                 (<= (- x3 4) (+ (* -2.0 x1) (* -2.0 x2)))
#                 )
#                 """
# 
#         bounds9 = (  [ -3,3], [ -3,3], [-1,5] )
#         vars9 = ['x1','x2','x3' ]
# 
#         F,X = plt.subplots(3,2)
#         # left: plot cuts for x3
#         ranges_left = { 'x1': bounds9[0], 'x2': bounds9[1] }
# 
#         ranges_left['x3'] = 0
#         parse_and_plot(ineq9,['x1','x2','x3'],ranges_left,axes=X[0,0],interactive=True)
#         ranges_left['x3'] = 1
#         parse_and_plot(ineq9,['x1','x2','x3'],ranges_left,axes=X[1,0],interactive=True)
#         ranges_left['x3'] = 2
#         parse_and_plot(ineq9,['x1','x2','x3'],ranges_left,axes=X[2,0],interactive=True)
#         
#         # right: plot cuts for x1
#         ranges_right = { 'x2': bounds9[1], 'x3': bounds9[2] }
# 
#         ranges_right['x1'] = 0
#         parse_and_plot(ineq9,['x1','x2','x3'],ranges_right,axes=X[0,1],interactive=True)
#         ranges_right['x1'] = 0.5
#         parse_and_plot(ineq9,['x1','x2','x3'],ranges_right,axes=X[1,1],interactive=True)
#         ranges_right['x1'] = 1.0
#         parse_and_plot(ineq9,['x1','x2','x3'],ranges_right,axes=X[2,1],interactive=True)
#         
#         plt.show()

        ###### tests for plotting SLICES
#         # 3D cube
#         ineq1 = """
#             (and
#                 (<= 0 x1)
#                 (<= x1 1)
#                 (<= 0 x2)
#                 (<= x2 1)
#                 (<= 0 x3)
#                 (<= x3 1)
#                 )
#                 """
# 
#         vars1 = [ 'x1','x2','x3' ]
#         basis1 = np.array([0.0,0.0,0.0])
#         dirs1a = np.array([ [1.0,1.0,0.0], [ 0.0,0.0,1.0 ] ])
#         dirs1b = np.array([ [1.0,0.5,0.0], [ 0.0,0.0,1.0 ] ])
# 
#         parsed_ineq1 = parse_smt2.parse_and_interpret(ineq1,vars1)[0]
# 
#         F,X = plt.subplots(1,2)
# 
#         plot_slice(parsed_ineq1,basis1,dirs1a,axes=X[0],x_bounds=[-3.0,3.0],y_bounds=[-1.0,2.0],axes_labels=('foo','bar'),interactive=True)
#         plot_slice(parsed_ineq1,basis1,dirs1b,axes=X[1],x_bounds=[-3.0,3.0],y_bounds=[-1.0,2.0],axes_labels=('foo2','bar2'),interactive=True)
# 
#         plt.suptitle('3D cube at origin')
#         plt.show()

#         # 3D cube moved
#         ineq1 = """
#             (and
#                 (<= 1 x1)
#                 (<= x1 2)
#                 (<= 0 x2)
#                 (<= x2 1)
#                 (<= 0.2 x3)
#                 (<= x3 1.2)
#                 )
#                 """
# 
#         vars1 = [ 'x1','x2','x3' ]
#         basis1 = np.array([0.0,0.0,0.0])
#         dirs1a = np.array([ [1.0,0.5,0.0], [ 0.0,0.0,1.0 ] ])
#         dirs1b = np.array([ [1.0,0.25,0.0], [ 0.0,0.0,1.0 ] ])
# 
#         parsed_ineq1 = parse_smt2.parse_and_interpret(ineq1,vars1)[0]
# 
#         F,X = plt.subplots(1,2)
# 
#         plot_slice(parsed_ineq1,basis1,dirs1a,axes=X[0],x_bounds=[-3.0,3.0],y_bounds=[-1.0,2.0],axes_labels=('foo','bar'),interactive=True)
#         plot_slice(parsed_ineq1,basis1,dirs1b,axes=X[1],x_bounds=[-3.0,3.0],y_bounds=[-1.0,2.0],axes_labels=('foo','bar'),interactive=True)
#         plt.show()

        # 3D cube moved, basis also moved
#         ineq1 = """
#             (and
#                 (<= 1 x1)
#                 (<= x1 2)
#                 (<= 0 x2)
#                 (<= x2 1)
#                 (<= 0.2 x3)
#                 (<= x3 1.2)
#                 )
#                 """
# 
#         vars1 = [ 'x1','x2','x3' ]
#         basis1 = np.array([1.0,0.0,0.0])
#         dirs1a = np.array([ [1.0,1.0,0.0], [ 0.0,0.0,1.0 ] ])
#         dirs1b = np.array([ [1.0,0.5,0.0], [ 0.0,0.0,1.0 ] ])
# 
#         parsed_ineq1 = parse_smt2.parse_and_interpret(ineq1,vars1)[0]
# 
#         F,X = plt.subplots(1,2)
# 
#         plot_slice(parsed_ineq1,basis1,dirs1a,axes=X[0],x_bounds=[-3.0,3.0],y_bounds=[-1.0,2.0],axes_labels=('foo','bar'),interactive=True)
#         plot_slice(parsed_ineq1,basis1,dirs1b,axes=X[1],x_bounds=[-3.0,3.0],y_bounds=[-1.0,2.0],axes_labels=('foo','bar'),interactive=True)
# 
#         plt.suptitle('3D cube moved, basis also moved')
#         plt.show()

        # Pyramide
        ineq2 = """
            (and
            (<= 0 x3)
            (<= (- x3 4) (+ (* 2.0 x1) (* 2.0 x2)))
            (<= (- x3 4) (+ (* -2.0 x1) (* 2.0 x2)))
            (<= (- x3 4) (+ (* 2.0 x1) (* -2.0 x2)))
            (<= (- x3 4) (+ (* -2.0 x1) (* -2.0 x2)))
            )
            """

        vars2 = [ 'x1','x2','x3' ]

        parsed_ineq2 = parse_smt2.parse_and_interpret(ineq2,vars2)[0]

        F,X = plt.subplots(1,3)

        # slice along (x2,x3) plane
        bas2a = np.array([0.0,0.0,0.0])
        dirs2a = np.array([[0.0,1.0,0.0],[0.0,0.0,1.0]])

        plot_slice(parsed_ineq2,bas2a,dirs2a,axes=X[0],x_bounds=[-4.0,4.0],y_bounds=[0.0,4.0],interactive=True)

        # slice along (x2,x3) plane, shifted
        bas2b = np.array([1.0,0.0,0.0])
        dirs2b = np.array([[0.0,1.0,0.0],[0.0,0.0,1.0]])

        plot_slice(parsed_ineq2,bas2b,dirs2b,axes=X[1],x_bounds=[-4.0,4.0],y_bounds=[0.0,4.0],interactive=True)

        # weird slice
        bas2c = np.array([1.0,0.0,0.5])
        dirs2c = np.array([[2.0,2.0,1.0],[2.0,-1.0,0]])

        plot_slice(parsed_ineq2,bas2c,dirs2c,axes=X[2],x_bounds=[-4.0,4.0],y_bounds=[-4.0,4.0],interactive=True)


        plt.show()

        

