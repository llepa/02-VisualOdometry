import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

def plot_estimated_world_points(file_path, save_path='estimated_world_points.png'):
    """
    Plot estimated world points in 3D.
    Each row in the file should contain:
        id x y z
    The id is ignored in the plot.
    """
    # Load data from file
    data = np.loadtxt(file_path)
    
    # Extract x, y, z components (columns 1, 2, 3)
    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Estimated World Points')
    
    # Save the figure
    plt.savefig(save_path)
    plt.close(fig)

def plot_errors(file_path, save_path='errors.png'):
    """
    Plot errors over iterations.
    Each row in the file should contain:
        iteration translational_error rotational_error
    Two lines (with different colors) are drawn for translational and rotational errors.
    """
    # Load data from file
    data = np.loadtxt(file_path)
    
    iterations = data[:, 0]
    translational_error = data[:, 1]
    rotational_error = data[:, 2]
    
    plt.figure()
    # Plot translational error
    plt.plot(iterations, translational_error, marker='o', color='r', label='Translational Error')
    # Plot rotational error
    plt.plot(iterations, rotational_error, marker='o', color='g', label='Rotational Error')
    
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Errors over Iterations')
    plt.legend()
    
    # Save the figure
    plt.savefig(save_path)
    plt.close()

def plot_trajectory(file_path, save_path='trajectory.png'):
    """
    Plot the trajectory.
    Each row in the file should contain:
        iteration estimated_x estimated_y ground_truth_x ground_truth_y
    The function connects consecutive points for both the estimated and the ground truth trajectories.
    """
    # Load data from file
    data = np.loadtxt(file_path)
    
    # Extract values
    # We don't need the iteration for plotting the trajectories (unless you want to use it to label the plot)
    estimated_x = data[:, 1]
    estimated_y = data[:, 2]
    ground_truth_x = data[:, 3]
    ground_truth_y = data[:, 4]
    
    plt.figure()
    # Plot estimated trajectory
    plt.plot(estimated_x, estimated_y, marker='o', color='b', label='Estimated Trajectory')
    # Plot ground truth trajectory
    plt.plot(ground_truth_x, ground_truth_y, marker='o', color='m', label='Ground Truth Trajectory')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectory Comparison')
    plt.legend()
    
    # Save the figure
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot 3D world points, error graphs, and trajectory data.')
    parser.add_argument('--world', type=str, required=True, help='Path to file with estimated world points.')
    parser.add_argument('--errors', type=str, required=True, help='Path to file with error data.')
    parser.add_argument('--trajectory', type=str, required=True, help='Path to file with trajectory data.')
    parser.add_argument('--output_prefix', type=str, default='plot', help='Prefix for the output image files.')
    args = parser.parse_args()
    
    # Create and save each plot with filenames based on the provided output prefix
    plot_estimated_world_points(args.world, f"{args.output_prefix}_world_points.png")
    plot_errors(args.errors, f"{args.output_prefix}_errors.png")
    plot_trajectory(args.trajectory, f"{args.output_prefix}_trajectory.png")

if __name__ == "__main__":
    main()
