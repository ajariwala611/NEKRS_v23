import numpy as np
from scipy.ndimage import uniform_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
import scipy.sparse as sp
import cvxpy as cp

def read_and_process_file(filename, nz, ny, nx):
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
        
        u_bar = data[:,0].reshape(nz, ny, nx,order='F')
        u_bar = np.mean(u_bar, axis=0)
        u_bar = np.tile(u_bar[np.newaxis, :, :], (nz, 1, 1))
        v_bar = data[:,1].reshape(nz, ny, nx,order='F')
        v_bar = np.mean(v_bar, axis=0)
        v_bar = np.tile(v_bar[np.newaxis, :, :], (nz, 1, 1))
        u_prime = data[:,2].reshape(nz, ny, nx,order='F')
        return u_bar,v_bar,u_prime
    
    except Exception as e:
        print(f"Error reading or processing the file: {e}")
        raise

def read_and_process_control_grid(filename, nz, ny, nx):
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
        v_prime = data.reshape(nz, ny, nx,order='F')
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

def predict_LSMs(u_bar, v_bar, u_prime, dt, num_steps, x_grid_meas, x_grid_pred, y_grid_meas, y_grid_pred):
    """
    Predict LSM states considering both x and y velocity components.

    Parameters:
        u_bar (numpy.ndarray): Mean velocity field in x-direction, size (nz, ny, nx_meas).
        v_bar (numpy.ndarray): Mean velocity field in y-direction, size (nz, ny, nx_meas).
        u_prime (numpy.ndarray): LSM field at the last time step, size (nz, ny, nx_meas).
        dt (float): Time step for prediction.
        num_steps (int): Number of future time steps to predict.
        x_grid_meas (numpy.ndarray): Measurement grid x-coordinates.
        x_grid_pred (numpy.ndarray): Prediction grid x-coordinates.
        y_grid_meas (numpy.ndarray): Measurement grid y-coordinates.
        y_grid_pred (numpy.ndarray): Prediction grid y-coordinates.

    Returns:
        numpy.ndarray: Predicted LSM data in the prediction grid, size (nz, ny_pred, nx_pred, num_steps).
    """
    # Extract dimensions
    nz, ny_meas, nx_meas = u_bar.shape
    ny_pred = len(y_grid_pred)
    nx_pred = len(x_grid_pred)

    x_start_pred = x_grid_pred[0]  # Start of prediction grid range
    y_start_pred = y_grid_pred[0]  # Start of y-grid prediction range

    dx_pred = x_grid_pred[1] - x_grid_pred[0]  # X-spacing in prediction grid
    dy_pred = y_grid_pred[1] - y_grid_pred[0]  # Y-spacing in prediction grid

    # Initialize prediction grid data (zeros initially)
    prediction_data = np.zeros((nz, ny_pred, nx_pred, num_steps))

    # Precompute displacement factors for all time steps
    time_displacements = np.arange(num_steps) * dt

    # Vectorized computation over all z and y indices
    for z in range(nz):
        for y in range(ny_meas):
            # Compute physical positions for each x_meas over time steps
            # Compute physical positions for each x_meas over time steps
            displacement_x = u_bar[z, y, :, np.newaxis] * time_displacements[np.newaxis, :]
            displacement_y = v_bar[z, y, :, np.newaxis] * time_displacements[np.newaxis, :]
            print(displacement_y.shape)
            new_x_physical = x_grid_meas[:, np.newaxis] + displacement_x  # Shape: (nx_meas, num_steps)
            new_y_physical = y_grid_meas[:, np.newaxis] + displacement_y  # Shape: (nx_meas, num_steps)

            # Identify valid indices within the prediction grid range
            valid_mask_x = (new_x_physical >= x_start_pred) & (new_x_physical <= x_grid_pred[-1])
            valid_mask_y = (new_y_physical >= y_start_pred) & (new_y_physical <= y_grid_pred[-1])

            # Compute lower and upper indices and weights
            lower_indices_x = np.clip(((new_x_physical - x_start_pred) // dx_pred).astype(int), 0, nx_pred - 1)
            lower_indices_y = np.clip(((new_y_physical - y_start_pred) // dy_pred).astype(int), 0, ny_pred - 1)
            upper_indices_x = np.clip(lower_indices_x + 1, 0, nx_pred - 1)
            upper_indices_y = np.clip(lower_indices_y + 1, 0, ny_pred - 1)
            lower_weights_x = (x_grid_pred[upper_indices_x] - new_x_physical) / dx_pred
            lower_weights_y = (y_grid_pred[upper_indices_y] - new_y_physical) / dy_pred
            upper_weights_x = 1.0 - lower_weights_x
            upper_weights_y = 1.0 - lower_weights_y

            # Apply valid mask to filter out invalid contributions
            lower_indices_x[~valid_mask_x] = 0
            upper_indices_x[~valid_mask_x] = 0
            lower_weights_x[~valid_mask_x] = 0
            upper_weights_x[~valid_mask_x] = 0
            lower_indices_y[~valid_mask_y] = 0
            upper_indices_y[~valid_mask_y] = 0
            lower_weights_y[~valid_mask_y] = 0
            upper_weights_y[~valid_mask_y] = 0

            # Accumulate weighted contributions
            for t in range(num_steps):
                np.add.at(prediction_data[z, y, :, t], lower_indices_x[:, t], lower_weights_x[:, t] * u_prime[z, y])
                np.add.at(prediction_data[z, y, :, t], upper_indices_x[:, t], upper_weights_y[:, t] * u_prime[z, y])
                np.add.at(prediction_data[z, y, :, t], lower_indices_y[:, t], lower_weights_y[:, t] * u_prime[z, y])
                np.add.at(prediction_data[z, y, :, t], upper_indices_y[:, t], upper_weights_y[:, t] * u_prime[z, y])

    return prediction_data

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
    nz_meas, ny_meas, nx_meas = u_prime.shape
    ny_pred, nx_pred = len(y_grid_pred), len(x_grid_pred)

    # Initialize the predicted scalar field storage
    prediction_data = np.zeros((nz_meas, ny_pred, nx_pred, num_steps))

    # Define an interpolator for the initial scalar field in 3D
    interpolator = RegularGridInterpolator(
        (z_grid_meas, y_grid_meas, x_grid_meas), 
        u_prime, 
        bounds_error=False, fill_value=0
    )

    # Generate a meshgrid for the prediction grid
    Z_pred, Y_pred, X_pred = np.meshgrid(z_grid_meas, y_grid_pred, x_grid_pred, indexing='ij')

    # Flatten for vectorized computation
    Z_pred_flat = Z_pred.ravel(order='F')
    Y_pred_flat = Y_pred.ravel(order='F')
    X_pred_flat = X_pred.ravel(order='F')
    # print(f'Max:',np.max(X_pred_flat),'Min:',np.min(X_pred_flat))
    # Time evolution
    for t in range(num_steps):
        # Compute backtracked positions (where the scalar value came from)
        # X_pred_flat = X_pred.ravel(order='F') - u_bar.ravel(order='F') * dt * (t+1) 
        X_pred_flat -= u_bar.ravel(order='F') * dt
        # Y_pred_flat -= v_bar.ravel(order='F') * dt

        # Stack the coordinates properly for interpolation
        points = np.vstack((Z_pred_flat, Y_pred_flat, X_pred_flat)).T  # Shape: (N_points, 3)

        # Interpolate values from measurement grid onto prediction grid
        u_prime_pred = interpolator(points).reshape(nz_meas, ny_pred, nx_pred,order='F')

        # Store results in the correct shape (nz, ny, nx, num_steps)
        prediction_data[:, :, :, t] = u_prime_pred

    return prediction_data


def solve_QP_J1(A, B, C, Q, R, x0, y_des, N, u0, u_min, u_max):
    """
    Solve the quadratic programming problem for the control system.

    Parameters:
        A, B, C (np.ndarray): State-space matrices of the system.
        Q, R (np.ndarray): Cost matrices for states and controls.
        x0 (np.ndarray): Initial state of the system.
        y_des (np.ndarray): Desired trajectory for output.
        N (int): Time horizon for prediction.
        u_min, u_max (float): Control input bounds.
        u0 (np.ndarray): Initial control input.

    Returns:
        np.ndarray: Optimal control sequence.
    """
    n, m = B.shape  # State and control dimensions
    # _, p = C.shape  # Output dimension

    # Define decision variable
    U = cp.Variable((m * N, 1))  # Control inputs stacked over N steps

    # Define matrices to stack system dynamics in vectorized form
    A_powers = [np.linalg.matrix_power(A, i) for i in range(N)]
    A_stack = sp.vstack([sp.csr_matrix(A_powers[i]) for i in range(N)])
    B_stack = sp.lil_matrix((n * N, m * N))
    for i in range(N):
        for j in range(i + 1):
            B_stack[i * n:(i + 1) * n, j * m:(j + 1) * m] = A_powers[i - j] @ B
    B_stack = sp.csr_matrix(B_stack)

    # Construct C_stack for outputs
    # C_stack = sp.kron(sp.eye(N), C, format="csr")

    # State trajectory
    x_trajectory = A_stack @ x0.reshape(n, 1) + B_stack @ U

    # Output trajectory
    # y_trajectory = x_trajectory
    # y_trajectory = C_stack @ x_trajectory

    # Desired trajectory flattened 
    y_flat = y_des.flatten().reshape(-1, 1)

    # Sparse cost matrices
    Q_sparse = sp.kron(sp.eye(N), Q*np.eye(n), format="csr")
    R_sparse = sp.kron(sp.eye(N), R*np.eye(m), format="csr")

    # Cost function
    cost = 0.5 * cp.quad_form(x_trajectory - y_flat, Q_sparse) + 0.5 * cp.quad_form(U, R_sparse)

    # Constraints
    constraints = []
    constraints.append(U[0:m] == u0)  # Initial control input
    U_min = np.full((m * N, 1), u_min)  # Minimum control input (e.g., -1 for each time step)
    U_max = np.full((m * N, 1), u_max)   # Maximum control input (e.g., 1 for each time step)
    constraints.append(U >= U_min)
    constraints.append(U <= U_max)

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    if problem.status != cp.OPTIMAL:
        raise ValueError(f"QP Solver did not converge. Status: {problem.status}")

    # Return the optimal control sequence
    return U.value.reshape(m, N)


if __name__ == "__main__":
    # Define the dimensions of the data
    nz, ny, nx = 21, 21, 101  # Replace with actual dimensions
    # Path to the input file
    lsm_filename = "observation_grid_data.txt"
    
    # Process the file
    u_bar,v_bar,u_prime = read_and_process_file(lsm_filename, nz, ny, nx)
    print(f" Read data from file: ",lsm_filename)
    u_prime_filtered = apply_convolution_filter(u_prime,5)
    # Debug: Print the reshaped data shape
    print(f" Applied cnvolution filter to u_prime")

    # Measurement Grid: Uniform straight box, the data is in -8 angle box
    x_grid_meas = np.linspace(0.5, 0.6, nx)
    y_grid_meas = np.linspace(0.094, 0.114, ny)
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
    predicted_LSM = predict_scalar_field(u_prime_filtered, u_bar, v_bar, dt_predict, N, x_grid_meas, x_grid_pred, y_grid_meas, y_grid_meas,z_grid_meas)
    print(" Predicted LSMs")

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
    R = 0.1   # Control cost weight (penalizes control effort)
    P = 1 # Terminal cost weight
    # x0 = C.T @ u_prime_filtered.reshape(p,1)  # Initial reducded order state
    # u0 = 0 # Inititla control input, should read that from last QP for continuity
    ctrl_grd_filename = "control_grid_data.txt"
    u0, v_prime = read_and_process_control_grid(ctrl_grd_filename, nx, ny, nz)
    if u0 > 1e-6:
        u0 = 1
    print(f" Read data from file: ",ctrl_grd_filename)
    x0 = C.T @ v_prime.reshape(p,1) # Initial reducded order state
    x0 = np.zeros((n,1))  # Initial state
    
    lamda = 100 # some weight for inducded desried downwash
    y_des = lamda * C.T @ predicted_LSM.reshape(p,N) # Desired downwash in reducded order state
    
    pi_min = 0.0 # Lower bound for control
    pi_max = 1.0 # Upper bound for control
    
    U_optimal = solve_QP_J1(A, B, C, Q, R, x0, y_des, N, u0, pi_min, pi_max)

    np.savetxt('u_optimal.txt',U_optimal.flatten())
    print(' Solved QP and saved the optimal control sequence in a file')