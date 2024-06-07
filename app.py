from flask import Flask, jsonify, request, render_template
import numpy as np

app = Flask(__name__)

# Gaussian elimination functions
def gaussian_elimination_inverse(A):
    n = len(A)
    if n != len(A[0]):
        return "Matrix is not square"

    # Augment A with the identity matrix
    A = np.array(A, dtype=float)
    I = np.identity(n)
    AugmentedMatrix = np.hstack((A, I))

    # Forward elimination to get REF
    for i in range(n):
        # Partial pivoting
        maxRow = i
        for k in range(i+1, n):
            if abs(AugmentedMatrix[k][i]) > abs(AugmentedMatrix[maxRow][i]):
                maxRow = k
        # Swap rows i and maxRow in AugmentedMatrix
        AugmentedMatrix[[i, maxRow]] = AugmentedMatrix[[maxRow, i]]

        # Make all rows below this one 0 in the current column
        for k in range(i+1, n):
            c = AugmentedMatrix[k][i] / AugmentedMatrix[i][i]
            AugmentedMatrix[k, i:] = AugmentedMatrix[k, i:] - c * AugmentedMatrix[i, i:]

    # Back substitution to transform into RREF
    for i in range(n-1, -1, -1):
        for k in range(i-1, -1, -1):
            c = AugmentedMatrix[k][i] / AugmentedMatrix[i][i]
            AugmentedMatrix[k, i:] = AugmentedMatrix[k, i:] - c * AugmentedMatrix[i, i:]

    # Normalize rows to get identity on the left side
    for i in range(n):
        divisor = AugmentedMatrix[i][i]
        AugmentedMatrix[i, i:] = AugmentedMatrix[i, i:] / divisor

    # Extract the inverse matrix from the augmented matrix
    InverseMatrix = AugmentedMatrix[:, n:]

    return [InverseMatrix, InverseMatrix.shape]

def GaussianEliminationSolve(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(A)
    m = len(A[0])
    
    # Augmented matrix
    AugmentedMatrix = np.hstack([A, b.reshape(-1, 1)])

    # Forward elimination to get REF
    for i in range(min(n, m)):
        # Partial pivoting
        maxRow = i
        for k in range(i + 1, n):
            if abs(AugmentedMatrix[k][i]) > abs(AugmentedMatrix[maxRow][i]):
                maxRow = k
        AugmentedMatrix[[i, maxRow]] = AugmentedMatrix[[maxRow, i]]  # Swap rows

        # Make all rows below this one 0 in the current column
        for k in range(i + 1, n):
            if AugmentedMatrix[i][i] == 0:
                continue  # Avoid division by zero
            c = AugmentedMatrix[k][i] / AugmentedMatrix[i][i]
            AugmentedMatrix[k, i:] = AugmentedMatrix[k, i:] - c * AugmentedMatrix[i, i:]

    # Back substitution to solve for the unknowns
    x = np.zeros(m)
    for i in range(min(n, m) - 1, -1, -1):
        if AugmentedMatrix[i][i] == 0:
            if AugmentedMatrix[i][m] != 0:
                return "No solutions"
            else:
                continue  # Free variable, can set to any value (handle underdetermined systems)
        x[i] = AugmentedMatrix[i][m] / AugmentedMatrix[i][i]
        for k in range(i - 1, -1, -1):
            AugmentedMatrix[k][m] -= AugmentedMatrix[k][i] * x[i]
    
    # Round each element of x to 2 decimal places
    rounded_x = np.round(x, decimals=2)
    print(rounded_x)
    return rounded_x

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve/solution', methods=['POST'])
def solveSolution():
    matrix_input = request.form['matrix']
    vector_input = request.form['vector']

    # Parse matrix input
    matrix_rows = matrix_input.strip().split(';')
    A = [list(map(float, row.split(','))) for row in matrix_rows]

    # Parse vector input
    b = list(map(float, vector_input.split(',')))

    # Solve the system
    solution = GaussianEliminationSolve(A, b)

    # Convert solution to string and return
    return jsonify(solution=solution.tolist())
    
@app.route('/solve/inverse', methods=['POST'])
def solveInverse():
    matrix_input = request.form['matrix']
    # Parse matrix input
    matrix_rows = matrix_input.strip().split(';')
    A = [list(map(float, row.split(','))) for row in matrix_rows]

    # Solve the system
    solution = gaussian_elimination_inverse(A)
    print(solution)
    return jsonify(solution=solution[0].tolist(),shape=solution[1])

if __name__ == '__main__':
    app.run(debug=True)
