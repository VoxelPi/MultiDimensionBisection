import numpy as np
from numpy.typing import NDArray

# Generates the n-complete matrix M_n.
def generate_complete_matrix_star(n: int):
    return np.array([ [ (i >> j) & 1 for j in range(n) ] for i in range(2**n) ])

def generate_initial_polygon(x0: NDArray[np.floating], h: NDArray[np.floating]):
    n = x0.shape[0]
    Ms = generate_complete_matrix_star(n)
    G = np.tile(x0, (2**n, 1))
    B = np.diag(h)
    return G + Ms @ B

def vector_sign(F, x: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.sign(F(x))

def bisection3_1d(f, x0: float, h: float, epsillon: float) -> float:
    x_n = x0
    required_iterations = int(np.ceil(np.log2(h/epsillon)))
    for n in range(required_iterations):
        x_n_1 = x_n
        x_n = x_n_1 + np.sign(f(x0)) * np.sign(f(x_n_1)) * h / 2**(n+1)

    # Check that an root has been found.
    if min(abs(x_n - x0), abs(x_n - x0 - h)) <= epsillon:
        return None
    return x_n

# Algorithm 3.2 (Creating an admissible n-polygon)
def construct_admissible_polygon(F, epsillon: float, x0: NDArray[np.floating], h: NDArray[np.floating]) -> NDArray[np.floating]:
    # Step 1.
    n = x0.shape[0]

    # Step 2.
    M = generate_complete_matrix_star(n) * 2 - 1
    
    # Step 3.
    R = generate_initial_polygon(x0, h)
    # print(f"Initial polygon:\n{R}")

    # Step 4.
    A = np.zeros(shape=(2**n, n))
    found_states = set()

    # Step 5.
    for R_i in R:
        sign = vector_sign(F, R_i)
        if (sign == 0).any(): # TODO: Check
            print(f"WARNING sign=0 for row {R_i}")
            continue

        # print(f"{R_i=} {sign=}")
        # A[(M == sign).all(axis=1)] = R_i
        type_index = np.where((M == sign).all(axis=1))[0][0]
        if type_index not in found_states:
            A[type_index, np.arange(n)] = R_i
            found_states.add(type_index)
    # print(f"Initial states: {found_states}")
    # print(f"Test:\n{A}")

    # Step 6.
    # Check if the matrix R has any row that is all 0
    if (not (A==0).all(axis=1).any()):
        # Step 23.
        return R
    
    # Step 7.
    # Find the orders (indices?) of the extreme points for all n * 2^(n-1) proper 1-simplexes S_beta = (S_beta^1, S_beta^2).
    proper_1_simplices = []
    for i in range(2**n):
        r_i = R[i]
        sign_x_i = np.sign(r_i)
        # sign_F_i = vector_sign(F, r_i)

        for j in range(i + 1, 2**n):

            r_j = R[j]
            sign_x_j = np.sign(r_j)
            # sign_F_j = vector_sign(F, r_j)

            # Search for a component that changes between the sign vectors i and j,
            k_sign_change = None
            for k in range(n):
                if (sign_x_i[k] == sign_x_j[k]):
                    continue
                else:
                    if k_sign_change is None:
                        # No change found until now, save k and look if there are other sign changes.
                        k_sign_change = k
                        continue
                    else:
                        # A sign change has already happened, so this simplex is not proper.
                        k_sign_change = None
                        break
            
            # If the 1-simplex is proper k_sign_change is set to the index where the change in sign happens.
            # Otherwise k_sign_change is set to None, and the next simplex should be checked.
            if k_sign_change is None:
                continue

            proper_1_simplices.append((i, j, k_sign_change))
    # print(proper_1_simplices)

    for (i, j, k_sign_change) in proper_1_simplices:
        r_i = R[i]
        r_j = R[j]
        x0 = min(r_i[k_sign_change], r_j[k_sign_change])
        h = abs(r_i[k_sign_change] - r_j[k_sign_change])
        for i_range in range(n):
            def f(x: float) -> float:
                v = np.copy(r_i)
                v[k_sign_change] = x
                return F(v)[i_range]
            x_r = bisection3_1d(f, x0, h, epsillon)
            if x_r is None:
                continue

            # print(f"Found a root of f_{i_range} at edge {i+1}-{j+1}: {x_r}")

            v_a = np.copy(r_i)
            v_a[k_sign_change] = x_r + 0.1
            sign_a = vector_sign(F, v_a)
            type_index_a = np.where((M == sign_a).all(axis=1))[0][0]
            if type_index_a not in found_states:
                A[type_index_a, np.arange(n)] = v_a
                found_states.add(type_index_a)
                # print(f"Found new position type: {type_index_a}")

            v_b = np.copy(r_i)
            v_b[k_sign_change] = x_r - 0.1
            sign_b = vector_sign(F, v_b)
            type_index_b = np.where((M == sign_b).all(axis=1))[0][0]
            if type_index_b not in found_states:
                A[type_index_b, np.arange(n)] = v_b
                found_states.add(type_index_b)
                # print(f"Found new position type: {type_index_b}")

    if (len(found_states) != 2**n):
        print("ERROR: Unable to find all possible states.")

    return A

# Bisection algorithm 3.
def bisection3(F, epsilon = 1e-9):
    # Form the intial system

    return np.nan

def F2(x: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.array([x[0]**2 - 4*x[1], x[1]**2 - 2*x[0] + 4*x[1]])

A = construct_admissible_polygon(F2, 1e-2, np.array([-2, -0.25]), np.array([4, 0.5]))
print(A)