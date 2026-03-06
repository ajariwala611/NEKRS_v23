import numpy as np
from scipy.ndimage import uniform_filter
from scipy.interpolate import RegularGridInterpolator
import scipy.sparse as sp
import cvxpy as cp

def read_and_process_file(filename, nx, ny, nz):
    """
    Reads data from a formatted text file, processes it into a numpy array,
    and reshapes it into a 3D array (nz, ny, nx).
    
    Args:
        filename (str): Path to the input text file.
        nz (int): Size of the first dimension.
        ny (int): Size of the second dimension.
        nx (int): Size of the third dimension.

    Returns:
        np.ndarray: Reshaped 3D numpy array with shape (nz, ny, nx, 2) for <u> and u_prime.
    """
    try:
        with open(filename, 'r') as file:
            # Skip the first two comment lines
            file.readline()
            file.readline()
            
            # Read the rest of the data into a numpy array
            data = np.loadtxt(file)
        # Ensure the total number of elements matches the expected dimensions
        total_elements = nz * ny * nx
        if data.shape[0] != total_elements:
            raise ValueError("Mismatch between data size and expected dimensions (nz, ny, nx).")
        
        # Reshape the data into (nz, ny, nx, 2) for <u> and u_prime
        
        u_bar = data[:,0].reshape(nx, ny, nz)
        u_bar = np.mean(u_bar, axis=2)
        u_bar = np.tile(u_bar[:, :, np.newaxis], (1, 1, nz))
        v_bar = data[:,1].reshape(nx, ny, nz)
        v_bar = np.mean(v_bar, axis=2)
        v_bar = np.tile(v_bar[:, :, np.newaxis], (1, 1, nz))
        u_prime = data[:,2].reshape(nx, ny, nz)
        return u_bar,v_bar,u_prime
    
    except Exception as e:
        print(f"Error reading or processing the file: {e}")
        raise

def read_and_process_control_grid(filename, nx, ny, nz):
    """
    Reads data from a formatted text file, processes it into a numpy array,
    and reshapes it into a 3D array (nz, ny, nx).
    
    Args:
        filename (str): Path to the input text file.
        nz (int): Size of the first dimension.
        ny (int): Size of the second dimension.
        nx (int): Size of the third dimension.

    Returns:
        np.ndarray: Reshaped 3D numpy array with shape (nz, ny, nx, 2) for <u> and u_prime.
    """
    try:
        with open(filename, 'r') as file:
            # Skip the first two comment lines
            file.readline()
            u0 = float(file.readline())
            file.readline()
            
            # Read the rest of the data into a numpy array
            data = np.loadtxt(file)
        # Ensure the total number of elements matches the expected dimensions
        total_elements = nz * ny * nx
        if data.shape[0] != total_elements:
            raise ValueError("Mismatch between data size and expected dimensions (nz, ny, nx).")
        
        # Reshape the data into (nz, ny, nx, 2) for <u> and u_prime
        v_prime = data.reshape(nx, ny, nz)
        return u0, v_prime
    
    except Exception as e:
        print(f"Error reading or processing the file: {e}")
        raise

def apply_convolution_filter(u_prime, kernel_size):
    """
    Applies a uniform convolution filter to u_prime with periodic boundary conditions along z-axis.

    Args:
        u_prime (np.ndarray): Input array of shape (nz, ny, nx).
        kernel_size (int): Size of the uniform filter kernel (assumed to be odd).

    Returns:
        np.ndarray: Filtered array of the same shape as u_prime.
    """
    return uniform_filter(u_prime, size=(kernel_size, kernel_size, kernel_size), mode='wrap')

def rotate_points(x, y, z, axis, angle, origin):
    """
    Rotate points (x, y, z) around the specified axis by the given angle.
    The rotation is performed around the provided origin.
    """
    # Shift points to the origin
    x_shifted = x - origin[0]
    y_shifted = y - origin[1]
    z_shifted = z - origin[2]
    angle_rad = np.radians(angle)  # Convert angle to radians
    c, s = np.cos(angle_rad), np.sin(angle_rad)

    if axis == 'x':
        R =  np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])
    elif axis == 'y':
        R =  np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    elif axis == 'z':
        R =  np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])


    # Stack points into a (3, N) array for matrix multiplication
    points = np.vstack((x_shifted, y_shifted, z_shifted))

    # Apply rotation
    rotated_points = R @ points

    # Shift points back to the original coordinate system
    x_rotated = rotated_points[0] + origin[0]
    y_rotated = rotated_points[1] + origin[1]
    z_rotated = rotated_points[2] + origin[2]

    return x_rotated, y_rotated, z_rotated

def predict_scalar_field(u_prime, u_bar, v_bar, dt, num_steps, 
                         x_grid_meas, x_grid_pred, y_grid_meas, y_grid_pred, 
                         z_grid_meas):
    """
    Predicts the evolution of a 3D scalar field (u_prime) in a moving velocity field 
    while interpolating onto a prediction grid. Movement occurs only in x- and y-directions.

    Parameters:
        u_prime (numpy.ndarray): Initial scalar field, shape (nz_meas, ny_meas, nx_meas)
        u_bar (numpy.ndarray): Mean velocity in x-direction, shape (nz_meas, ny_meas, nx_meas)
        v_bar (numpy.ndarray): Mean velocity in y-direction, shape (nz_meas, ny_meas, nx_meas)
        dt (float): Time step for prediction
        num_steps (int): Number of future time steps to predict
        x_grid_meas (numpy.ndarray): Measurement grid x-coordinates
        x_grid_pred (numpy.ndarray): Prediction grid x-coordinates
        y_grid_meas (numpy.ndarray): Measurement grid y-coordinates
        y_grid_pred (numpy.ndarray): Prediction grid y-coordinates
        z_grid_meas (numpy.ndarray): Measurement grid z-coordinates (unchanged in prediction)

    Returns:
        numpy.ndarray: Predicted scalar field, shape (nz_meas, ny_pred, nx_pred, num_steps)
    """
    # Get grid dimensions
    nx_meas, ny_meas, nz_meas = u_prime.shape
    ny_pred, nx_pred = len(y_grid_pred), len(x_grid_pred)

    # Initialize the predicted scalar field storage
    prediction_data = np.zeros((nx_pred, ny_pred, nz_meas, num_steps))

    # Define an interpolator for the initial scalar field in 3D
    interpolator = RegularGridInterpolator(
        (x_grid_meas, y_grid_meas, z_grid_meas), 
        u_prime, 
        bounds_error=False, fill_value=0, method="linear"
    )
    #Flip U_bar and V_bar for correct index
    # u_bar += u_prime
    u_bar = np.flip(u_bar,axis=0)
    v_bar = np.flip(v_bar,axis=0)
    
    # Generate a meshgrid for the prediction grid
    X_pred, Y_pred, Z_pred = np.meshgrid(x_grid_pred, y_grid_pred, z_grid_meas, indexing='ij')
    origin_pred = (X_pred.min(), Y_pred.min(), (Z_pred.min() + Z_pred.max())/2)
    origin_meas = (x_grid_meas.min(), y_grid_meas.min(), (z_grid_meas.min() + z_grid_meas.max())/2)
    
    X_pred, Y_pred, Z_pred = rotate_points(X_pred.flatten(), Y_pred.flatten(), Z_pred.flatten(), axis='z', angle=-8, origin=origin_pred)

    # Flatten for vectorized computation
    Z_pred_flat = Z_pred
    Y_pred_flat = Y_pred
    X_pred_flat = X_pred
    
    # Time evolution
    for t in range(num_steps):
        # Compute backtracked positions (where the scalar value came from)
        X_pred_flat -= u_bar.ravel() * dt
        Y_pred_flat -= v_bar.ravel() * dt

        #rotating data to convert into local coordinates for interpolation
        X_pred_flat,Y_pred_flat,Z_pred_flat = rotate_points(X_pred_flat, Y_pred_flat, Z_pred_flat, 'z', 6, origin_meas)

        # Stack the coordinates properly for interpolation
        points = np.vstack((X_pred_flat, Y_pred_flat, Z_pred_flat)).T  # Shape: (N_points, 3)

        # Interpolate values from measurement grid onto prediction grid
        u_prime_pred = interpolator(points).reshape(nx_pred, ny_pred, nz_meas)

        #rotating back to convert into global coordinates for interpolation
        X_pred_flat,Y_pred_flat,Z_pred_flat = rotate_points(X_pred_flat, Y_pred_flat, Z_pred_flat, 'z', -6, origin_meas)

        # Store results in the correct shape (nz, ny, nx, num_steps)
        prediction_data[:, :, :, t] = u_prime_pred

    return prediction_data

def solve_QP_J1(A, B, C, Q, R, x0, y_des, N, u0, u_min, u_max):
    n, m = B.shape
    U = cp.Variable((m * (N), 1))  # Stacked control inputs over N-1 time steps

    # Define matrices to stack system dynamics in vectorized form
    A_stack = sp.lil_matrix((n * N, n))  # Stack of A matrices (using LIL for efficient construction)
    B_stack = sp.lil_matrix((n * N, m * N))  # Stack of B matrices

    # Construct A_stack and B_stack
    for i in range(N):
        A_stack[i*n:(i+1)*n, :] = np.linalg.matrix_power(A, i+1)
        for j in range(i+1):
            B_stack[i*n:(i+1)*n, j*m:(j+1)*m] = np.linalg.matrix_power(A, i-j).dot(B)

    A_stack = sp.csr_matrix(A_stack)  # Convert to CSR for efficient arithmetic operations
    B_stack = sp.csr_matrix(B_stack)

    x_trajectory = A_stack.dot(x0.reshape(n, 1)) + B_stack.dot(U)
    y_trajectory = x_trajectory  # Keeping trajectory in reduced order state
    
    # Compute C_stack for outputs y = C * x
    # C_stack = sp.kron(sp.eye(N), C)  # Kronecker product to repeat C for each step
    # C_stack = sp.csr_matrix(C_stack)
    # y_trajectory = C_stack.dot(x_trajectory)  # Shape: (p * N, 1)

    # Flatten the reference trajectory and reshape it to match y_trajectory's shape
    r_flattened = y_des.reshape(n*N, 1)  # Reshape to (p*(N-1), 1)
    Q_sparse = sp.kron(sp.eye(N), Q*sp.eye(n))  # Sparse version of the cost matrix
    R_sparse = sp.kron(sp.eye(N), R*sp.eye(m))  # Sparse version of the control matrix
    # Running cost: sum of quadratic forms for (y_k - r_k)^T Q (y_k - r_k) + u_k^T R u_k
    running_cost = 0.5 * cp.quad_form(y_trajectory - r_flattened, Q_sparse) + \
                0.5 * cp.quad_form(U, R_sparse)

    # Compute the final state x(N)
    # x_N = np.linalg.matrix_power(A_tilde, N) @ x0 + B_stack[-n:, :] @ U  # Shape: (n, 1)
    # Terminal cost
    # y_N = C @ x_N
    # terminal_cost = 0.5 * cp.quad_form(y_N - r_N, P)

    # Total cost running + terminal
    # total_cost = running_cost + terminal_cost

    # Define the total cost: running
    total_cost = running_cost 

    constraints = []
    epsilon = 0.01  # Tolerance
    # constraints.append(cp.norm(y_N - yref) <= epsilon)
    constraints.append(U[0:m] == u0)  # Initial control input
    U_min = np.full((m * N, 1), u_min)  # Minimum control input (e.g., -1 for each time step)
    U_max = np.full((m * N, 1), u_max)   # Maximum control input (e.g., 1 for each time step)
    constraints.append(U >= U_min)
    constraints.append(U <= U_max)

    # Formulate the optimization problem with the final state constraint
    problem = cp.Problem(cp.Minimize(total_cost), constraints)

    # Solve the optimization problem
    problem.solve()
    if problem.status != cp.OPTIMAL:
        raise ValueError(f"QP Solver did not converge. Status: {problem.status}")

    # Extract the optimal control input vector
    return U.value


if __name__ == "__main__":
    # Define the dimensions of the data
    nx, ny, nz = 101, 21, 21  # Replace with actual dimensions
    # Path to the input file
    lsm_filename = "observation_grid_data.txt"
    
    # Process the file
    u_bar,v_bar,u_prime = read_and_process_file(lsm_filename, nx, ny, nz)
    print(f"Read data from file: ",lsm_filename)
    u_prime_filtered = apply_convolution_filter(u_prime,3)
    # Debug: Print the reshaped data shape
    print(f"Applied cnvolution filter to u_prime")

    # Measurement Grid: Uniform straight box, the data is in -8 angle box
    x_grid_meas = np.linspace(0.5, 0.6, nx)
    y_grid_meas = np.linspace(0.0925, 0.114, ny)
    z_grid_meas = np.linspace(0.025, 0.075, nz)
    # Z_meas, Y_meas, X_meas = np.meshgrid(z_grid_meas, y_grid_meas, x_grid_meas, indexing='ij')

    # Prediction Grid: Uniform straight box
    x_grid_pred = np.linspace(0.6, 0.7, nx) # To convect the LSM in control only
    y_grid_pred = np.linspace(0.082, 0.102, ny)
    z_grid_pred = np.linspace(0.025, 0.075, nz)
    # Z_pred, Y_pred, X_pred = np.meshgrid(z_grid_pred, y_grid_pred, x_grid_pred, indexing='ij')

    N = 50 # Time horizona for prediction and control problem
    dt_predict = 1e-3 # Time step for prediction and control problem
    # predicted_LSM = predict_LSMs(u_bar, v_bar, u_prime_filtered, dt_predict, N, x_grid_meas, x_grid_pred, y_grid_meas, y_grid_pred)
    predicted_LSM = predict_scalar_field(u_prime_filtered, u_bar, v_bar, 1e-3, N, x_grid_meas, x_grid_pred, y_grid_meas, y_grid_pred, z_grid_meas)
    print("Predicted LSMs")

    # Read A,B, and C matrices
    # A = pd.read_csv('A_out.txt',header=None,sep='\s+').to_numpy()
    # B = pd.read_csv('B_out.txt',header=None,sep='\s+').to_numpy()
    # C = pd.read_csv('C.txt',header=None,sep='\s+').to_numpy()
    A = np.loadtxt('A_out.txt')
    B = np.loadtxt('B_out.txt').reshape(-1, 1)
    C = np.loadtxt('C.txt')
    n = A.shape[0] # Dimensions of the reduced order state
    m = B.shape[1] # Number of controller
    p = C.shape[0] # Number of full state points

    # Solve the SDP
    Q = 100  # Output cost weight (penalizes output deviation)
    R = 1   # Control cost weight (penalizes control effort)
    P = 1 # Terminal cost weight
    # x0 = C.T @ u_prime_filtered.reshape(p,1)  # Initial reducded order state
    # u0 = 0 # Inititla control input, should read that from last QP for continuity
    ctrl_grd_filename = "control_grid_data.txt"
    u0, v_prime = read_and_process_control_grid(ctrl_grd_filename, nx, ny, nz)
    print(f"Read data from file: ",ctrl_grd_filename)
    x0 = C.T @ v_prime.reshape(p,1) # Initial reducded order state
    x0 = np.zeros((n,1))  # Initial state
    
    lamda = -500 # some weight for inducded desried downwash
    y_des = lamda * C.T @ predicted_LSM.reshape(p,N) # Desired downwash in reducded order state
    
    pi_min = 0.0 # Lower bound for control
    pi_max = 1.0 # Upper bound for control
    
    U_optimal = solve_QP_J1(A, B, C, Q, R, x0, y_des, N, u0, pi_min, pi_max)
    u_filtered = np.where(np.abs(U_optimal) < 1e-2, 0, 1)
    np.savetxt('u_optimal.txt',u_filtered.flatten(), fmt='%.4f')
    print('Solved QP and saved the optimal control sequence in a file')