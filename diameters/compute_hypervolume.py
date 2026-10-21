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
import random

import parse_smt2
import math



def plot(ineq,fixed_vars,x_bounds,y_bounds,*,axes=None,interactive=False,axes_labels=(None,None)):
    assert ineq.shape[1] - 3 == fixed_vars.shape[0]

    fixed_part = np.dot(ineq[:,2:-1],fixed_vars)[:,None]

    reduced_ineq = np.concatenate([ineq[:,:2],fixed_part + ineq[:,-1:]],1)
    return plot_2d(reduced_ineq,x_bounds,y_bounds,axes=axes,interactive=interactive,axes_labels=axes_labels)


def plot_slice(ineq,slice_basis,slice_dirs,x_bounds,y_bounds,*,axes=None,interactive=False,axes_labels=(None,None)):
   
    slice_dirs_norm = slice_dirs / np.linalg.norm(slice_dirs,ord=2,axis=1,keepdims=True)

  
    projected_ineq = np.full((ineq.shape[0],3),np.nan,dtype=ineq.dtype)
    for pos in range(ineq.shape[0]):
        this_ineq = ineq[pos]
        proj0 = np.dot(this_ineq[:-1],slice_dirs_norm[0])
        proj1 = np.dot(this_ineq[:-1],slice_dirs_norm[1])
        projc = np.dot(this_ineq[:-1],slice_basis) + this_ineq[-1]

        projected_ineq[pos] = np.array([proj0,proj1,projc])

    return plot_2d(projected_ineq,x_bounds,y_bounds,axes=axes,interactive=interactive,axes_labels=axes_labels)


def plot_2d(ineq,x_bounds,y_bounds,*,axes=None,interactive=False,axes_labels=(None,None)):
    x = np.linspace(x_bounds[0],x_bounds[1],num=1000)
    y = np.linspace(y_bounds[0],y_bounds[1],num=1000)
    grid = np.meshgrid(x,y)
    const = np.ones_like(grid[0])
    pts = np.stack([grid[0],grid[1],const],axis=-1).reshape((-1,3))


    ev = pts @ ineq.T
    sat = (ev >= -0.1)
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



def parse_and_plot(ineq,all_vars,ranges,*,axes=None,interactive=False):
    parsed_ineq,parsed_vars = parse_smt2.parse_and_interpret(ineq,all_vars)

    plotted_vars = []
    fixed_vars = []

    for var in all_vars:
        try:
            val = float(ranges[var])
            fixed_vars.append(var)
        except TypeError:
            plotted_vars.append(var)

    assert len(plotted_vars) == 2

    x_bounds = ranges[plotted_vars[0]]
    y_bounds = ranges[plotted_vars[1]]

    fixed_values = [ ranges[f] for f in fixed_vars ]


    new_variable_order = [ all_vars.index(p) for p in plotted_vars ] + [ all_vars.index(f) for f in fixed_vars ] + [ -1 ]
    reordered_ineq = parsed_ineq[:,new_variable_order]

    return plot(reordered_ineq,np.array(fixed_values),x_bounds,y_bounds,axes=axes,interactive=interactive,axes_labels=(plotted_vars[0],plotted_vars[1]))
    




num_samples=100000

def hypervolume(ineq,all_vars,ranges):
    parsed_ineq,parsed_vars = parse_smt2.parse_and_interpret(ineq,all_vars)
    
    high = []
    low = []
    Vol = 1.0
    for i in range(len(all_vars)):
        high.append(ranges[all_vars[i]][1])
        low.append(ranges[all_vars[i]][0])
        Vol *= (high[i]-low[i])
    
    sat_points = 0
    
    for simulation in range(num_samples):

        sample_point = []
        cond = True
        for i in range(len(all_vars)):
            sample_point.append(random.random()*(high[i]-low[i])+low[i])
        
        for i in range(parsed_ineq.shape[0]):
            sum = 0.0
            for j in range(parsed_ineq.shape[1]-1):
                sum+=sample_point[j]*parsed_ineq[i,j]
            if (sum<-parsed_ineq[i,parsed_ineq.shape[1]-1]):
                cond=False
        
        if (cond):
            sat_points+=1
    
    h = sat_points/num_samples*Vol
    print("Estimated hypervolume: "+str(h))
    

    
def hypervolume_clever(ineq,all_vars,ranges):
    parsed_ineq,parsed_vars = parse_smt2.parse_and_interpret(ineq,all_vars)
    
    high = []
    low = []
    Vol = 1.0
    for i in range(len(all_vars)):
        high.append(ranges[all_vars[i]][1])
        low.append(ranges[all_vars[i]][0])
        if (high[i]-low[i]!=0.0):
            Vol *= (high[i]-low[i])
    
    sat_points = 0
    
    for simulation in range(num_samples):

        sample_point = []
        cond = True
        for i in range(len(all_vars)): 
            sample_point.append(random.random()*(high[i]-low[i])+low[i])
        
        for i in range(parsed_ineq.shape[0]):
            sum = 0.0
            for j in range(parsed_ineq.shape[1]-1):
                sum+=sample_point[j]*parsed_ineq[i,j]
            if (sum<-parsed_ineq[i,parsed_ineq.shape[1]-1]):
                cond=False
        
        if (cond):
            sat_points+=1
    
    h = sat_points/num_samples*Vol
    print("Estimated hypervolume: "+str(h))
    

    
def max_distance(ineq,all_vars,ranges):
    parsed_ineq,parsed_vars = parse_smt2.parse_and_interpret(ineq,all_vars)
    
    high = []
    low = []
    Vol = 1.0
    for i in range(len(all_vars)):
        high.append(ranges[all_vars[i]][1])
        low.append(ranges[all_vars[i]][0])
        Vol *= (high[i]-low[i])
    
    sat_points = 0
        
    good_points = []
    
    for simulation in range(num_samples):

        sample_point = []
        cond = True
        for i in range(len(all_vars)):
            sample_point.append(random.random()*(high[i]-low[i])+low[i])
        
        for i in range(parsed_ineq.shape[0]):
            sum = 0.0
            for j in range(parsed_ineq.shape[1]-1):
                sum+=sample_point[j]*parsed_ineq[i,j]
            if (sum<-parsed_ineq[i,parsed_ineq.shape[1]-1]):
                cond=False
        
        if (cond):
            sat_points+=1
            good_points.append(sample_point)
            
   
    dist = 0.0
    best_direction = []
    for i in range(len(good_points)):
        for j in range(len(good_points)):
            sum = 0.0
            for k in range(len(good_points[i])):
                sum += (good_points[i][k]-good_points[j][k])*(good_points[i][k]-good_points[j][k])
            current_dist = math.sqrt(sum)
            if (current_dist>dist):
                dist = current_dist
                best_direction = []
                for l in range(len(good_points[i])):
                    best_direction.append(good_points[i][l]-good_points[j][l])

    print("# Points evaluated: "+str(len(good_points))+" out of "+str(num_samples))
    print("Maximum distance found: "+str(dist))
    print("Direction of maximum diameter found: "+str(best_direction))
    
    
def max_distance_on_surface(ineq,all_vars,ranges):
    parsed_ineq,parsed_vars = parse_smt2.parse_and_interpret(ineq,all_vars)
    
    high = []
    low = []
    Vol = 1.0
    for i in range(len(all_vars)):
        high.append(ranges[all_vars[i]][1])
        low.append(ranges[all_vars[i]][0])
        Vol *= (high[i]-low[i])
    
    sat_points = 0
        
    good_points = []
    
    for simulation in range(math.floor(num_samples/2)):

        sample_point = []
        cond = True
        for i in range(len(all_vars)):
            sample_point.append(random.random()*(high[i]-low[i])+low[i])
        
        a = random.randint(0,parsed_ineq.shape[0]-1)
        b = random.randint(0,len(all_vars)-1)
        
        while (parsed_ineq[a,b]==0.0):
            a = random.randint(0,parsed_ineq.shape[0]-1)
            b = random.randint(0,len(all_vars)-1)
        
        sum2 = 0.0
        for j in range(parsed_ineq.shape[1]-1):
            if (j!=b):
                sum2+=sample_point[j]*parsed_ineq[a,j]
        
        sample_point[b] = (-parsed_ineq[a,parsed_ineq.shape[1]-1]-sum2)/parsed_ineq[a,b]
        
        for i in range(parsed_ineq.shape[0]):
            sum = 0.0
            for j in range(parsed_ineq.shape[1]-1):
                sum+=sample_point[j]*parsed_ineq[i,j]
            if (sum<-parsed_ineq[i,parsed_ineq.shape[1]-1]):
                cond=False
        
        if (cond):
            sat_points+=1
            good_points.append(sample_point)
            
    dist = 0.0
    best_direction = []
    for i in range(len(good_points)):
        for j in range(len(good_points)):
            sum = 0.0
            for k in range(len(good_points[i])):
                sum += (good_points[i][k]-good_points[j][k])*(good_points[i][k]-good_points[j][k])
            current_dist = math.sqrt(sum)
            if (current_dist>dist):
                dist = current_dist
                best_direction = []
                for l in range(len(good_points[i])):
                    best_direction.append(good_points[i][l]-good_points[j][l])

    print("# Points evaluated: "+str(len(good_points))+" out of "+str(math.floor(num_samples/2)))
    print("Maximum distance found: "+str(dist))
    print("Direction of maximum diameter found: "+str(best_direction))
    
def max_distance_exact2D(ineq,all_vars,ranges,output_file):
    tollerance_error = -10e-06
    parsed_ineq,parsed_vars = parse_smt2.parse_and_interpret(ineq,all_vars)
    
    if (parsed_ineq.shape[1]!=3):
        print("ERROR: This method cannot be applied for object in more than 2 dimensions")
        return
    
    sat_points = 0
    tot_points = 0
    good_points = []
    
    
    high = []
    low = []
    for i in range(len(all_vars)):
        high.append(ranges[all_vars[i]][1])
        low.append(ranges[all_vars[i]][0])
        
    for i in range(len(high)):
        a1 = 0
        a2 = 0
        if (i==0):
            a1 = -1
        if (i==1):
            a2 = -1
        borders_ineq = np.array([[a1,a2,high[i]]])
        parsed_ineq = np.concatenate((parsed_ineq,borders_ineq))

    for i in range(len(low)):
        a1 = 0
        a2 = 0
        if (i==0):
            a1 = 1
        if (i==1):
            a2 = 1
        borders_ineq = np.array([[a1,a2,low[i]]])
        parsed_ineq = np.concatenate((parsed_ineq,borders_ineq))
        
    
    for y in range(parsed_ineq.shape[0]):
        for z in range(parsed_ineq.shape[0]):
            if (y>z and parsed_ineq[y,0]*parsed_ineq[z,1]-parsed_ineq[y,1]*parsed_ineq[z,0]!=0):
                cond = True
                sample_point = [(parsed_ineq[y,1]*parsed_ineq[z,2]-parsed_ineq[y,2]*parsed_ineq[z,1])/(parsed_ineq[y,0]*parsed_ineq[z,1]-parsed_ineq[y,1]*parsed_ineq[z,0]),
                                (parsed_ineq[y,2]*parsed_ineq[z,0]-parsed_ineq[y,0]*parsed_ineq[z,2])/(parsed_ineq[y,0]*parsed_ineq[z,1]-parsed_ineq[y,1]*parsed_ineq[z,0])]
                
                tot_points+=1
                
                for i in range(parsed_ineq.shape[0]):
                    sum = 0.0
                    for j in range(parsed_ineq.shape[1]-1):
                        sum+=sample_point[j]*parsed_ineq[i,j]
                    if (sum+parsed_ineq[i,parsed_ineq.shape[1]-1]<tollerance_error):
                        cond=False
                
                if (cond):
                    sat_points+=1
                    good_points.append(sample_point)
        
    dist = 0.0
    best_direction = []
    best_points = []
    for i in range(len(good_points)):
        for j in range(len(good_points)):
            sum = 0.0
            for k in range(len(good_points[i])):
                sum += (good_points[i][k]-good_points[j][k])*(good_points[i][k]-good_points[j][k])
            current_dist = math.sqrt(sum)
            if (current_dist>dist):
                dist = current_dist
                best_direction = []
                for l in range(len(good_points[i])):
                    best_direction.append(good_points[i][l]-good_points[j][l])
                    best_points = [good_points[i],good_points[j]]
                    
    print("# Points evaluated: "+str(len(good_points))+" out of "+str(tot_points))
    print("Maximum distance found: "+str(dist))
    print("Direction of maximum diameter found: "+str(best_direction))
    print("Vertices: "+str(best_points))
    
    P1_x = best_points[0][0]
    P1_y = best_points[0][1]
    P2_x = best_points[1][0]
    P2_y = best_points[1][1]
    
    delta_x = P2_x - P1_x
    delta_y = P2_y - P1_y
    
    name1 = all_vars[0]
    name2 = all_vars[1]
    
    smt_formula = "(and\n"
    if (delta_x!=0):
        new_value_coeff = delta_y/delta_x
        my_custom_shift = new_value_coeff * (-P2_x) + P2_y
        
        if (my_custom_shift>=0):
            smt_formula += "\t(= "+name2+" (+ (* "+str(new_value_coeff)+" "+name1+") "+str(abs(my_custom_shift))+"))\n"
        else:
            smt_formula += "\t(= "+name2+" (- (* "+str(new_value_coeff)+" "+name1+") "+str(abs(my_custom_shift))+"))\n"
        smt_formula += "\t(>= "+name1+" "+str(min(best_points[0][0],best_points[1][0]))+") \n"
        smt_formula += "\t(<= "+name1+" "+str(max(best_points[1][0],best_points[0][0]))+") \n"
        smt_formula +=")\n"
        print()
        print()
        print(smt_formula)
    else:
        smt_formula += "\t(= "+name1+" "+str(P1_x)+")\n"
        smt_formula += "\t(>= "+name2+" "+str(min(P1_y,P2_y))+")\n"
        smt_formula += "\t(<= "+name2+" "+str(max(P2_y,P1_y))+")\n"
        smt_formula += ")\n"
        print()
        print()
        print(smt_formula)
         
    with open(output_file + ".smt2", "w") as g:
        g.write(smt_formula)



        


def hypervolume_shoelace_exact2D(ineq,all_vars,ranges):
    parsed_ineq,parsed_vars = parse_smt2.parse_and_interpret(ineq,all_vars)
    
    if (parsed_ineq.shape[1]!=3):
        print("ERROR: This method cannot be applied for object in more than 2 dimensions")
        return



if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Plot a polytope, given as set of inequalities or smt2 file.')
    argparser.add_argument('-t','--test',action='store_true')
    argparser.add_argument('--filename',nargs='?')

    argparser.add_argument('--slices',nargs='*',help='Values of extra variables (to plot a slice), in form "x4=1.234 x11=5.678"')
    
    argparser.add_argument('--out', '-o', required=False,
                   help='output path without suffix (writes .smt2)')
    args = argparser.parse_args()
    if not args.test:
        # needs filename
        if args.filename is None :
            print('If --test is not given, the positional arguments (filename and variable bounds) must be given')
            argparser.print_help()
            sys.exit()
        data = open(args.filename,'rt').read()
        data = ''.join(data.splitlines())



        slices = {}
        variables = []
        for pos,sl in enumerate(args.slices):
            var,bound = sl.split('=')

            bound = [ float(b) for b in bound.split(',') ]
            slices[var] = bound
            variables.append(var)


        animate = False # TODO
    
        print()
    
        print()
     
        
        print()
     
        
        print()
        print("METHOD #3: exact method, all distances between vertices")
        max_distance_exact2D(data,variables,slices,args.out)
      

    else:
      
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


        bas2a = np.array([0.0,0.0,0.0])
        dirs2a = np.array([[0.0,1.0,0.0],[0.0,0.0,1.0]])

        plot_slice(parsed_ineq2,bas2a,dirs2a,axes=X[0],x_bounds=[-4.0,4.0],y_bounds=[0.0,4.0],interactive=True)


        bas2b = np.array([1.0,0.0,0.0])
        dirs2b = np.array([[0.0,1.0,0.0],[0.0,0.0,1.0]])

        plot_slice(parsed_ineq2,bas2b,dirs2b,axes=X[1],x_bounds=[-4.0,4.0],y_bounds=[0.0,4.0],interactive=True)

        bas2c = np.array([1.0,0.0,0.5])
        dirs2c = np.array([[2.0,2.0,1.0],[2.0,-1.0,0]])

        plot_slice(parsed_ineq2,bas2c,dirs2c,axes=X[2],x_bounds=[-4.0,4.0],y_bounds=[-4.0,4.0],interactive=True)


        plt.show()

        

