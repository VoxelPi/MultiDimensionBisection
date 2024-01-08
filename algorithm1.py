import numpy as np

# Returns the sign of the given function on the given interval between a and b.
# If the sign of the function varies between a and b, nan is returned.
# signeval in paper/89
def interval_sign(f, a, b, density = 100):
    x = np.linspace(start=a, stop=b, num=density) #int(np.ceil(interval_density * (b - a)))
    fx = np.vectorize(f)(x)
    min = np.min(fx)
    max = np.max(fx)
    # print(f"min={min} max={max}") # Debug interval f(x)

    if min >= 0:
        return 1
    elif max <= 0:
        return -1
    else:
        return np.nan

# K = ((a_1, b_1), (a_2, b_2))

# Checks if the given box satisifies the P.M. requirement.
def satisfies_PM(F, K):
    (I_1, I_2) = K
    (a_1, b_1) = I_1
    (a_2, b_2) = I_2

    # Calculate sign along the four edges.
    sign_f11 = interval_sign(lambda x_1 : F(np.array([x_1, a_2]))[1], a_1, b_1)
    sign_f12 = interval_sign(lambda x_1 : F(np.array([x_1, b_2]))[1], a_1, b_1)
    sign_f21 = interval_sign(lambda x_2 : F(np.array([a_1, x_2]))[0], a_2, b_2)
    sign_f22 = interval_sign(lambda x_2 : F(np.array([b_1, x_2]))[0], a_2, b_2)

    # Check P.M. condition
    return (sign_f11*sign_f12 <= 0) and (sign_f21*sign_f22 <= 0)

# Calculates the center of the box.
def refinement_center(K_0):
    (I_1, I_2) = K_0
    (a_1, b_1) = I_1
    (a_2, b_2) = I_2
    return np.array([(a_1 + b_1) / 2, (a_2 + b_2) / 2])

# Generates the refinement of the given box.
def generate_refinements(K):
    # Get interval from box.
    (I_1, I_2) = K
    (a_1, b_1) = I_1
    (a_2, b_2) = I_2

    # Calculate center.
    c_1 = (a_1 + b_1) / 2
    c_2 = (a_2 + b_2) / 2

    # Generate refinements.
    K1 = ((a_1, c_1), (a_2, c_2))
    K2 = ((c_1, b_1), (a_2, c_2))
    K3 = ((a_1, c_1), (c_2, b_2))
    K4 = ((c_1, b_1), (c_2, b_2))
    return [K1, K2, K3, K4]

# Bisection algorithm 1.
def bisection1(F, dF, K_0, delta=1e-6, max_iterations = 1000):
    # Check that the initial rectangle satisfies the P.M. condition.
    if not satisfies_PM(F, K_0):
        print("Invalid initial configuration")
        return np.nan
    
    center = refinement_center(K_0)
    error = np.linalg.norm(F(center), np.inf)
    iteration = 0
    tries = 0
    G = F
    K = K_0

    while error > delta and tries < 2 and iteration < max_iterations:
        iteration += 1
        tries += 1
        # print(f"Start of iteration {iteration}")
        refinements = generate_refinements(K)
        for refinement in refinements:
            if satisfies_PM(G, refinement):
                center = refinement_center(refinement)
                K = refinement
                error = np.linalg.norm(F(center), np.inf)
                tries -= 1
                # print(f"  Found refinement {index}")
                break # Go to next iteration.

        # Precondition
        dF_center = dF(center)
        dF_inv_center = np.linalg.inv(dF_center)
        G = lambda x : dF_inv_center @ F(x) # noqa: E731
        # print("  Applied preconditioning.")

    print(f"Found root: iteratin={iteration} tries={tries} error={error}")
    return center
