import sys

import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from sympy.matrices.expressions.blockmatrix import bounds
import parse_smt2


def project_subspace(ineqs, eqs, bounds, projection_direction=[0, 1]):
    """
    Project a convex subspace defined by Ax <= b onto the a-b plane using linear programming.
    :param A: Coefficients for the inequalities
    :param b: Constants for the inequalities
    :param bounds: The bounds for the variables
    """
    # Set up storage for projected points
    projected_points = []

    A_eq, b_eq = eqs
    A_ub, b_ub = ineqs

    # Iterate through angles to approximate the boundary
    theta = np.linspace(0, 2 * np.pi, 1000)  # Angle resolution for the projection boundary
    for angle in theta:
        # Define a direction vector in terms of a and b
        direction = [0 for i in range(len(bounds))]
        direction[projection_direction[0]] = np.cos(angle)
        direction[projection_direction[1]] = np.sin(angle)

        # Solve for the maximum value in the given direction
        result = linprog(-np.array(direction), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')
        if not result.success:
            # print(f"Failed for angle {angle}: {result.message}")
            continue
        if result.success:
            # print(result.x)
            projected_points.append([result.x[i] for i in projection_direction])  # Only take (a, b) components

    # Convert projected points to a numpy array for plotting
    # print(f"Number of points in projection in direction X{projection_direction[0]}, X{projection_direction[1]}:  ", len(projected_points))
    projected_points = np.array(projected_points)

    if len(projected_points) == 0:
        # print("No solution found for projection in direction ", projection_direction)
        return None

    return projected_points


def plot_projection(projected_points, title, projection_direction):
    # Plot the projection
    plt.figure(figsize=(8, 6))
    plt.fill(projected_points[:, 0], projected_points[:, 1], color='lightblue', alpha=0.7, label='Projected Convex Subspace')
    plt.scatter(projected_points[:, 0], projected_points[:, 1], color='black', label='Projected Points', s=10)  # Add dots
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title(title)
    plt.xlabel(f"X{projection_direction[0]}-axis")
    plt.ylabel(f"X{projection_direction[1]}-axis")
    plt.grid(alpha=0.3)
    plt.legend()
    # plt.show()


if __name__ == '__main__':
    HeartAttack = True
    input_size = 13

    datafile = "examples/itp_weak_ucore.smt2"
    data = open(datafile,'rt').read()
    data = data.splitlines()
    datapoint = data[0]
    # data = data[0]

    # each of these should be "x4=-3.14,2.75"
    vars = [f'x{i}' for i in range(1, 14)]

    parsed_ineq,parsed_vars = parse_smt2.parse_and_interpret(datapoint, vars)
    assert set(vars) == set(parsed_vars)
    parsed_ineq_A = parsed_ineq[:, :-1]  # Shape (5, input_size)
    parsed_ineq_b = parsed_ineq[:, -1].reshape(1, -1)


    lower_bounds = [29.0, 0.0, 0.0, 94.0, 126.0, 0.0, 0.0, 71.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    upper_bounds = [77.0, 1.0, 3.0, 200.0, 564.0, 1.0, 2.0, 202.0, 1.0, 6.2, 2.0, 4.0, 3.0]
    bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))]
    success, fail = 0, 0
    for a in range(input_size):
        for b in range(a):
            if a != b:
                # project_and_plot(parsed_ineq_A, parsed_ineq_b[0], [(0, None ) for i in range(14)], projection_direction=[a,b])
                # print(f"Exp #{ind} in direction X{a+1} and X{b+1}")
                projected_points = project_subspace(parsed_ineq_A, parsed_ineq_b[0],
                                                    bounds, projection_direction=[a, b])
                if projected_points is not None:
                    plot_projection(projected_points, projection_direction=[a+1,b+1], title=f"Projection in direction X{a+1} and X{b+1}")
                else:
                    print(f"Explanation projection on : X{a+1}, X{b+1} :  Failed!")


def project_multi_subspace(parsed_formulas, bounds, projection_direction):
    projected_spaces = []
    for parsed_ineq, parsed_eq in parsed_formulas:
        parsed_ineq_A = parsed_ineq[:, :-1]
        parsed_ineq_b = -1 * (parsed_ineq[:, -1].reshape(1, -1))
        parsed_eq_A = parsed_eq[:, :-1]
        parsed_eq_b = -1 * (parsed_eq[:, -1].reshape(1, -1))
        projected_points = project_subspace((parsed_ineq_A, parsed_ineq_b[0]), (parsed_eq_A, parsed_eq_b[0]),
                                                       bounds, projection_direction=projection_direction)
        # unique_lst = remove_redundant_values(projected_points)
        projected_spaces.append(projected_points)
    return projected_spaces


def remove_redundant_values(lst):
    if lst is None:
        return None
    unique_set = set(tuple(sublist) for sublist in lst)
    return [list(item) for item in unique_set]
