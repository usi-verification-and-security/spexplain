The program plot_polytope.py plots a polytope, given as a series of inequalities in CNF. Currently max two variables are supported, and they must have a name starting with "x".

Also plots the direction of highest variance in the polytope if desired (must set plot_pca in the script to "True"). Does not seem to be very useful.

Requirements: matplotlib, numpy, pyparsing, scikit-learn (only for PCA).

Example:
python plot_polytope.py itp_two_vars_weak_ucore.smt2 x1=0,1000 x3=-10,10

(note that the bounds must be given - it is not easy to infer good bounds from the data).

