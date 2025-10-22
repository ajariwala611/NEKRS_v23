import numpy as np

def rotation_matrix(axis, angle):
    """
    Create a 3D rotation matrix around the specified axis ('x', 'y', or 'z').
    Angle is in degrees.
    """
    angle_rad = np.radians(angle)  # Convert angle to radians
    c, s = np.cos(angle_rad), np.sin(angle_rad)

    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

def rotate_points(x, y, z, axis, angle, origin):
    """
    Rotate points (x, y, z) around the specified axis by the given angle.
    The rotation is performed around the provided origin.
    """
    # Shift points to the origin
    x_shifted = x - origin[0]
    y_shifted = y - origin[1]
    z_shifted = z - origin[2]

    # Create the rotation matrix
    R = rotation_matrix(axis, angle)

    # Stack points into a (3, N) array for matrix multiplication
    points = np.vstack((x_shifted, y_shifted, z_shifted))

    # Apply rotation
    rotated_points = R @ points

    # Shift points back to the original coordinate system
    x_rotated = rotated_points[0] + origin[0]
    y_rotated = rotated_points[1] + origin[1]
    z_rotated = rotated_points[2] + origin[2]

    return x_rotated, y_rotated, z_rotated

if __name__ == '__main__':
    # Define the grid coordinates
    x_ = np.linspace(0.5, 0.6, 101)
    y_ = np.linspace(0.094, 0.114, 21)
    z_ = np.linspace(0.025, 0.075, 21)

    # Generate the grid
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

    # Flatten the grid to obtain 1D arrays of points
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    # Define rotation parameters
    axis = 'z'  # Choose 'x', 'y', or 'z'
    angle = -6   # Rotation angle in degrees

    # Origin at the beginning of the box (minimum values of x_, y_, and z_)
    origin = (x_.min(), y_.min(), (z_.min() + z_.max())/2)

    # Rotate the points
    x_rotated, y_rotated, z_rotated = rotate_points(x, y, z, axis, angle, origin)

    # Write the rotated points to a file
    filename = 'observation_grid.txt'
    with open(filename, 'w') as f:
        f.write('%d\n' % len(x_rotated))
        for i in range(len(x_rotated)):
            f.write('%f %f %f\n' % (x_rotated[i], y_rotated[i], z_rotated[i]))
