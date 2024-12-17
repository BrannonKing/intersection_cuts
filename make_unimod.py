import numpy as np
import gurobipy as gp


def unimodular_lower_triangular(matrix: np.ndarray):
    n = matrix.shape[1]
    result = matrix.zeros((n, n))
    for col in range(n):
        result[col, col] = 1

        # max_val = row.max()
        # if max_val <= 0:
        #     result[]


    for col in range(m):
        # Step 1: Ensure pivot is nonzero (row swap if necessary)
        if matrix[col, col] == 0:
            for row in range(col + 1, m):
                if matrix[row, col] != 0:
                    matrix[[col, row]] = matrix[[row, col]]  # Swap rows
                    print(f"Swapped rows {col} and {row}") # :\n{matrix}\n")
                    break

        if matrix[col, col] != 1 and matrix[col, col] != -1:
            factor = matrix[col, col]
            matrix[col] //= factor  # Scale the entire row by the diagonal element
            print(f"Scaled row {col} by {factor}") # :\n{matrix}\n")
        
        # Step 2: Eliminate entries below the pivot
        for row in range(col + 1, m):
            if matrix[row, col] != 0:
                factor = -matrix[row, col] // matrix[col, col]  # Integer factor
                matrix[row] += factor * matrix[col]
                print(f"Added {factor} * row {col} to row {row}") # :\n{matrix}\n")
    
    return matrix


m = 5
n = 10
A = np.random.rand(m, n)
A[A < 0.3] = 0
B = np.ceil(A)
A[A > 0.65] = -1
A[A > 0.0] = 1
assert np.linalg.matrix_rank(A) == m

u = greedy_unimodular_transform(A)
print(A)
print(u)
print(A @ u)

# m2, u = make_rows_consistent(A)
# print(A)
# print(m2)
# print(u)
# model = gp.Model()
# x = model.addMVar(shape=(m, m), name='x')
# y = model.addMVar(shape=m, vtype=gp.GRB.BINARY, name='y')
# z = model.addMVar(shape=m, vtype=gp.GRB.BINARY, name='z')
# xA = x @ A
# for i in range(m):
#     model.addConstr(x[i, 0:i] == 0)  # upper triangular
#     model.addConstr(x[i, i] == 2*z[i] - 1)  # |det| = 1
#     model.addGenConstrIndicator(y[i], 1, xA[i, :] >= 0)
#     model.addGenConstrIndicator(y[i], 0, xA[i, :] <= 0)
#     # model.addGenConstrIndicator(y[i], 1, xA[i, :].sum() >= 0.1)
#     # for j in range(n):
#     #     model.addGenConstrIndicator(y[i], 1, B[i, j] <= x[i, :] @ A[:, j])

# model.setObjective(y.sum(), gp.GRB.MAXIMIZE)
# model.optimize()

# you have to do the whole thing, where you have more rows than columns
