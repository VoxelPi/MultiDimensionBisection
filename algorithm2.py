import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

# F(X) = F(x, y) = (f(x, y), g(x, y))

# (f(X_i), g(X_i)) wird (u_i, v_i), u_i = sign(f(X_i)), v_i = sign(g(X_i))

#             |   u_i   v_i |
# b_i = 1/8 * | u_i+1 v_i+1 | = 1/8 * (u_i*v_(i+1) - u_(i+1)*v_i)

# Approximation of the topological degree of F at theta
# d(f, D, theta) = sum_(i=1)^N b_i

def calculate_d(F, polygon):
    n_points = polygon.shape[0]

    u = np.zeros(n_points)
    v = np.zeros(n_points)
    for i in range(n_points):
        [f_x_i, g_x_i] = F(polygon[i])
        u[i] = np.sign(f_x_i)
        v[i] = np.sign(g_x_i)

    sum = 0
    for i in range(n_points):
        j = i+1 if i < n_points-1 else 0
        b_i = (u[i]*v[j] - u[j]*v[i]) / 8
        sum += b_i
    
    return sum
        
def triangulate_polygon(polygon: np.ndarray[float]) -> np.ndarray[float]:
    # tri = Delaunay(initial_boundary)
    # print(tri.simplices)
    # plt.triplot(initial_boundary[:,0], initial_boundary[:,1], tri.simplices)
    # plt.show()

    simplices = Delaunay(initial_boundary).simplices
    return np.array([(polygon[i_0], polygon[i_1], polygon[i_2]) for [i_0,i_1,i_2] in simplices])

# Bisection algorithm 2.
def bisection2(F, polygon: np.ndarray[float], epsilon = 1e-9):
    # STEP 1
    # Check if root is contained in the initial polygon
    d = calculate_d(F, polygon)
    if d == 0:
        return np.nan

    # STEP 2
    # Triangulate the polygon
    triangles = triangulate_polygon(polygon)
    n_triangles = triangles.shape[0]
    print(f"{n_triangles=}")

    def L(A, B, X):
        return (B[1] - A[1]) * (X[0] - A[0]) - (B[0] - A[0]) * (X[1] - A[1])
    ZERO = np.zeros((2))

    min_x = np.min(polygon[:,0])
    max_x = np.min(polygon[:,0])
    min_y = np.min(polygon[:,1])
    max_y = np.min(polygon[:,1])

    # Possible states
    # Normal: None
    # Bisected Triangle ("bisected"): (i_first_sub_triangle,original_triangle,i_longest_edge)
    # Parallelogram ("parallelogram"): (i_first_sub_triangle)

    while True:
        # STEP 3,4,5
        state = None
        state_data = None
        i_triangle = 0
        while i_triangle < n_triangles:
            # Get triangle vertices
            triangle = triangles[i_triangle]
            [x_1, x_2, x_3] = triangle
            # print(x_1, x_2, x_3)

            # Evaluate function F on triangle vertices.
            F_triangle = np.array([F(x_i) for x_i in triangle])
            [Fx_1, Fx_2, Fx_3] = F_triangle
            # print(Fx_1, Fx_2, Fx_3)

            # Check if the image of the triangle contains a root.
            contains_root = L(Fx_1, Fx_2, ZERO)*L(Fx_1, Fx_2, Fx_3) >= 0 and L(Fx_2, Fx_3, ZERO)*L(Fx_2, Fx_3, Fx_1) >= 0 and L(Fx_3, Fx_1, ZERO)*L(Fx_3, Fx_1, Fx_2) >= 0
            # print(f"{i_triangle=} {contains_root=}")
            if not contains_root:
                if state is None:
                    i_triangle += 1
                    continue

                elif state == "bisected":
                    # Check if first or second sub triangle failed.
                    # If first failed, check the second, otherwise continue with step 8.
                    (i_first_sub_triangle, triangle, i_max) = state_data
                    if i_triangle <= i_first_sub_triangle:
                        i_triangle += 1
                        continue

                    # STEP 8
                    # Construct paralellogram.
                    point_E = triangle[i_max] + triangle[(i_max + 1) % 3] - triangle[(i_max + 2) % 3]

                    # Check that the new point is in the boundary
                    if point_E[0] < min_x or point_E[0] > max_x or point_E[1] < min_y or point_E[1] > max_y:
                        # goto STATE 10
                        break

                    point_D = (triangle[i_max] + triangle[(i_max + 1) % 3]) / 2
                    sub_triangle_1 = np.array([triangle[i_max], point_E, point_D])
                    sub_triangle_2 = np.array([point_D, point_E, triangle[(i_max + 1) % 3]])

                    # Insert new triangles into list.
                    triangles = np.insert(triangles, i_triangle + 1, sub_triangle_1, axis=0)
                    triangles = np.insert(triangles, i_triangle + 2, sub_triangle_2, axis=0)
                    n_triangles += 2
                    state = "parallelogram"
                    state_data = (i_triangle + 1,)

                    i_triangle += 1
                    continue

                elif state == "parallelogram":
                    # Check if first or second sub triangle failed.
                    # If first failed, check the second, otherwise continue with step 8.
                    (i_first_sub_triangle,) = state_data
                    if i_triangle <= i_first_sub_triangle:
                        i_triangle += 1
                        continue

                    # Paralellogram failed, go to step 10 and bisect every triangle.
                    break

            # Root is contained in triangle.    
            # print(f"{i_triangle=} {contains_root=}")
            state = None
            state_data = None

            # STEP 7
            # Calculate the triangle edge lenghts and find the longest. 
            l_i = [ np.linalg.norm(triangle[i] - triangle[(i + 1) % 3], ord=2) for i in range(3) ]
            i_max = l_i.index(max(l_i))
            l_max = l_i[i_max]
            # print(f"  {i_max=} {l_i=}")

            # Check if tolerance is reached.
            if l_max < epsilon:
                return (triangle, F_triangle, l_max)

            # STEP 6
            # Bisect triangle.
            D = (triangle[i_max] + triangle[(i_max + 1) % 3]) / 2
            sub_triangle_1 = np.array([triangle[i_max], D, triangle[(i_max + 2) % 3]])
            sub_triangle_2 = np.array([D,triangle[(i_max + 1) % 3],  triangle[(i_max + 2) % 3]])

            # Insert new triangles into list.
            triangles[i_triangle] = sub_triangle_1
            triangles = np.insert(triangles, i_triangle + 1, sub_triangle_2, axis=0)
            n_triangles += 1
            state = "bisected"
            state_data = (i_triangle, triangle, i_max)

            continue # Do not increment triangle counter, as original triangle was replaced.

        # STATE 10
        # Bisect every triangle and start again.
        print(f"[STATE 10] Bisecting every triangle {n_triangles=}")
        new_triangles = []
        for triangle in triangles:
            # Calculate the triangle edge lenghts and find the longest. 
            l_i = [ np.linalg.norm(triangle[i] - triangle[(i + 1) % 3], ord=2) for i in range(3) ]
            i_max = l_i.index(max(l_i))
            l_max = l_i[i_max]

            # Generate bisected triangles.
            D = (triangle[i_max] + triangle[(i_max + 1) % 3]) / 2
            sub_triangle_1 = np.array([triangle[i_max], D, triangle[(i_max + 2) % 3]])
            sub_triangle_2 = np.array([D,triangle[(i_max + 1) % 3],  triangle[(i_max + 2) % 3]])
            new_triangles.append(sub_triangle_1)
            new_triangles.append(sub_triangle_2)
        triangles = np.array(new_triangles)
        n_triangles *= 2

    return np.nan

# Example function
def F(x):
    return np.array([x[0]**2 - 4*x[1], x[1]**2 - 2*x[0] + 4*x[1]])

initial_boundary = np.array([
    [-0.5,   0.25], #1
    [-2.0,   0.25], #2
    [-2.0,  -0.25], #3
    [ 0.75, -0.25], #4
    [ 2.0,  -0.25], #5
    [ 2.0,   0.25], #6
    [ 0.75,  0.25], #7
])

(x,f,l) = bisection2(F, initial_boundary, epsilon=0.001)
print(x)
print(f)
print(l)