import numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
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
        nz (int): Size of the third dimension.

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

def load_current_u_prime_ctrl(filename, shape):
    path = Path(filename)
    if not path.exists():
        print(f" {filename} not found; initialized current_u_prime_ctrl with zeros")
        return np.zeros(shape)

    current_u_prime_ctrl = np.load(path)
    if current_u_prime_ctrl.shape != shape:
        raise ValueError(
            f"{filename} has shape {current_u_prime_ctrl.shape}, expected {shape}."
        )

    print(f" Loaded current_u_prime_ctrl from {filename}")
    return current_u_prime_ctrl

def save_current_u_prime_ctrl(filename, current_u_prime_ctrl):
    path = Path(filename)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as file:
        np.save(file, current_u_prime_ctrl)
    tmp_path.replace(path)
    print(f" Saved current_u_prime_ctrl to {filename}")

def predict_scalar_field_with_indices(current_u_prime_obs, current_u_prime_ctrl, index_filename, num_steps):
    if current_u_prime_obs.shape != current_u_prime_ctrl.shape:
        raise ValueError(
            "current_u_prime_obs and current_u_prime_ctrl must have the same shape. "
            f"Got {current_u_prime_obs.shape} and {current_u_prime_ctrl.shape}."
        )

    with np.load(index_filename) as data:
        i_loc = data["i_loc"]
        j_loc = data["j_loc"]

    nx, ny, _ = current_u_prime_obs.shape
    if i_loc.shape != (num_steps, nx, ny) or j_loc.shape != (num_steps, nx, ny):
        raise ValueError(
            f"{index_filename} contains i_loc/j_loc shapes {i_loc.shape}/{j_loc.shape}, "
            f"expected {(num_steps, nx, ny)}."
        )

    obs_ctrl_LSM_stacked = np.vstack((current_u_prime_obs, current_u_prime_ctrl))
    if i_loc.min() < 0 or i_loc.max() >= obs_ctrl_LSM_stacked.shape[0]:
        raise ValueError(
            f"i_loc values must be in [0, {obs_ctrl_LSM_stacked.shape[0] - 1}], "
            f"got [{i_loc.min()}, {i_loc.max()}]."
        )
    if j_loc.min() < 0 or j_loc.max() >= obs_ctrl_LSM_stacked.shape[1]:
        raise ValueError(
            f"j_loc values must be in [0, {obs_ctrl_LSM_stacked.shape[1] - 1}], "
            f"got [{j_loc.min()}, {j_loc.max()}]."
        )

    predicted_LSM = obs_ctrl_LSM_stacked[i_loc, j_loc]
    return np.transpose(predicted_LSM, (1, 2, 3, 0))

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
    u_bar = np.flip(u_bar,axis=0)
    v_bar = np.flip(v_bar,axis=0)
    
    # Generate a meshgrid for the prediction grid
    X_pred, Y_pred, Z_pred = np.meshgrid(x_grid_pred, y_grid_pred, z_grid_meas, indexing='ij')
    origin_pred = (X_pred.min(), Y_pred.min(), (Z_pred.min() + Z_pred.max())/2)
    origin_meas = (x_grid_meas.min(), y_grid_meas.min(), (z_grid_meas.min() + z_grid_meas.max())/2)
    
    X_pred, Y_pred, Z_pred = rotate_points(X_pred.flatten(), Y_pred.flatten(), Z_pred.flatten(), axis='z', angle=-6, origin=origin_pred)

    # Flatten for vectorized computation
    Z_pred_flat = Z_pred
    Y_pred_flat = Y_pred
    X_pred_flat = X_pred

    # Time evolution
    for t in range(num_steps-1):
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
        prediction_data[:, :, :, t+1] = u_prime_pred

    return prediction_data

def solve_QP_J1(A, B, C, Q, R, x0, y_des, N, u0, u_min, u_max):
    n, m = B.shape

    x0 = np.asarray(x0).reshape(n, 1)
    y_des = np.asarray(y_des).reshape(n, N, order='F')

    U = cp.Variable((m * N, 1))

    A_stack = sp.lil_matrix((n * N, n))
    B_stack = sp.lil_matrix((n * N, m * N))

    for i in range(N):
        A_stack[i*n:(i+1)*n, :] = np.linalg.matrix_power(A, i+1)
        for j in range(i+1):
            B_stack[i*n:(i+1)*n, j*m:(j+1)*m] = np.linalg.matrix_power(A, i-j) @ B

    A_stack = sp.csr_matrix(A_stack)
    B_stack = sp.csr_matrix(B_stack)

    x_trajectory = A_stack @ x0 + B_stack @ U          # (nN,1)
    y_trajectory = x_trajectory                        # reduced-state tracking

    r_flattened = y_des.reshape(n * N, 1, order='F')  # MUST be F

    Q_sparse = sp.kron(sp.eye(N), Q * sp.eye(n))
    R_sparse = sp.kron(sp.eye(N), R * sp.eye(m))

    cost = 0.5 * cp.quad_form(y_trajectory - r_flattened, Q_sparse) \
         + 0.5 * cp.quad_form(U, R_sparse)

    constraints = [
        U[0:m] == np.asarray(u0).reshape(m, 1),
        U >= np.full((m * N, 1), u_min),
        U <= np.full((m * N, 1), u_max),
    ]

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    if problem.status != cp.OPTIMAL:
        raise ValueError(f"QP Solver did not converge. Status: {problem.status}")

    return U.value

if __name__ == "__main__":
    # Define the dimensions of the data
    nx, ny, nz = 101, 21, 21  # Replace with actual dimensions
    # Path to the input file
    lsm_filename = "observation_grid_data.txt"
    
    # Process the file
    _, _, u_prime = read_and_process_file(lsm_filename, nx, ny, nz)
    print(f" Read data from file: ",lsm_filename)
    u_prime_filtered = apply_convolution_filter(u_prime,3)
    # Debug: Print the reshaped data shape
    print(f" Applied cnvolution filter to u_prime")

    N = 100 # Time horizona for prediction and control problem
    N_apply = 25 # Number of predicted frames advanced between MPC calls
    index_filename = "taylor_indices_NACA4412_Re400k.npz"
    current_u_prime_ctrl_filename = "current_u_prime_ctrl.npy"

    current_u_prime_ctrl = load_current_u_prime_ctrl(
        current_u_prime_ctrl_filename, (nx, ny, nz)
    )
    predicted_LSM = predict_scalar_field_with_indices(
        u_prime_filtered, current_u_prime_ctrl, index_filename, N
    )
    next_u_prime_ctrl = predicted_LSM[:, :, :, N_apply - 1]
    print(" Predicted LSMs using precomputed i_loc/j_loc")

    A = np.loadtxt('A_out.txt')
    n = A.shape[0]  # Dimensions of the state and control
    B = np.loadtxt('B_out.txt').reshape(n, 1)
    m = B.shape[1]
    C = np.loadtxt('C.txt')
    p = C.shape[0]
    # Solve the SDP
    Q = 1.0  # Output cost weight (penalizes output deviation)
    R = 1.0   # Control cost weight (penalizes control effort)
    P = 1 # Terminal cost weight
    # x0 = C.T @ u_prime_filtered.reshape(p,1)  # Initial reducded order state
    # u0 = 0 # Inititla control input, should read that from last QP for continuity
    ctrl_grd_filename = "control_grid_data.txt"
    u0, v_prime = read_and_process_control_grid(ctrl_grd_filename, nx, ny, nz)
    print(f" Read data from file: ",ctrl_grd_filename)
    x0 = C.T @ v_prime.reshape(p,1,order='F') # Initial reducded order state
    
    lamda = -10 # some weight for inducded desried downwash
    y_des = lamda * C.T @ predicted_LSM.reshape(p,N,order='F') # Desired downwash in reducded order state
    
    pi_min = 0.0 # Lower bound for control
    pi_max = 1.0 # Upper bound for control
    
    U_optimal = solve_QP_J1(A, B, C, Q, R, x0, y_des, N, u0, pi_min, pi_max)

    u_filtered = np.where(np.abs(U_optimal) < 1e-2, 0, 1)
    np.savetxt('u_optimal.txt',u_filtered.flatten(), fmt='%.4f')
    save_current_u_prime_ctrl(current_u_prime_ctrl_filename, next_u_prime_ctrl)
    print(' Solved QP and saved the optimal control sequence in a file')
