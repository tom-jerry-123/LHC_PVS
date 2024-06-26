"""
Calculates and plots vertex density
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import uproot

ROOT_PATH = "C:/Users/jerry/Documents/Phys_Summer_Research/root_files/ttbar_hist-Rel21sample.root"


def get_tree():
    with uproot.open(ROOT_PATH) as file:
        tree_name = "EventTree;1"
        tree = file[tree_name]

    return tree


"""
Functions for plotting vertex count
"""

def plot_actual_count(bin_size, bin_range, show_plot=True):
    """
    Histogram of z-coordinates of vertices in an event. First event chosen
    :return:
    """
    tree = get_tree()
    vertex_z_array = tree['recovertex_z'].array()
    event_arr = vertex_z_array[0]

    # Create the bin edges
    bin_edges = np.arange(bin_range[0], bin_range[1] + bin_size, bin_size)

    # Create the histogram
    plt.hist(event_arr, bins=bin_edges, alpha=0.7, color='red', edgecolor='black', label="Actual")

    # Add labels and title
    plt.xlabel('Z-Coordinate')
    plt.ylabel('Vertex Count')
    plt.title('Vertex Count vs Z-Coordinate')

    # Show the plot
    if show_plot:
        plt.show()

    return len(event_arr)


def plot_expected_count(bin_size=1, bin_range=None, num_vertex=200, show_plot=True):
    # Define the mean and standard deviation, and number of vertices
    mean = 0
    std_dev = 45
    N = num_vertex

    # Default bin range
    if bin_range is None:
        bin_range = (mean - 3 * std_dev, mean + 3 * std_dev)

    # Create a range of x values
    x = np.arange(bin_range[0] + 0.5 * bin_size, bin_range[1] + 0.5 * bin_size, bin_size)

    # Calculate the Gaussian (normal) distribution values for y
    y = N * bin_size * norm.pdf(x, mean, std_dev)

    # Plot the Gaussian curve
    plt.plot(x, y, label='Expected')

    # Add a legend
    plt.legend()

    # Show the plot
    if show_plot:
        plt.show()


def vertex_count_plot(bin_size=10, bin_range=(-120, 120)):
    num_vertex = plot_actual_count(bin_size, bin_range, show_plot=False)
    plot_expected_count(bin_size, bin_range, num_vertex, show_plot=False)

    # Set plot labels, Show the plot
    plt.xlabel("Z-Coordinate")
    plt.ylabel("Vertex Density")
    plt.title("Vertex Density vs. Z-Coordinate")
    plt.show()


"""
Functions for plotting vertex density
"""


def plot_actual_density(bin_size, bin_range, event_num=0, show_plot=True):
    tree = get_tree()
    vertex_z_array = tree['recovertex_z'].array()
    event_arr = vertex_z_array[event_num]

    # Create the bin edges
    bin_edges = np.arange(bin_range[0], bin_range[1] + bin_size, bin_size)

    # get histogram info
    bin_edges = np.arange(bin_range[0], bin_range[1] + bin_size, bin_size)
    frequencies, edges = np.histogram(event_arr, bins=bin_edges)

    # Compute density = bin_count / bin_size
    density = frequencies / bin_size
    # Compute midpoints of edges. Not needed for step plot
    # midpoints = (edges[:-1] + edges[1:]) / 2

    # Add information to plot
    plt.step(edges[:-1], density, where='mid', color='red', label='Actual')
    plt.xlabel('Z-Coordinate')
    plt.ylabel('Vertex Density')

    # Show the plot
    if show_plot:
        plt.show()

    return len(event_arr)


def vertex_density_plot(bin_size, bin_range, event_num=0):
    num_vertex = plot_actual_density(bin_size, bin_range, event_num=event_num, show_plot=False)
    # Note: when bin_size==1, expected count for bin and density numerically equal
    # Assumption: standard deviation of Z-coordinate spread remains about 45.
    plot_expected_count(bin_size=1, bin_range=bin_range, num_vertex=num_vertex, show_plot=False)

    # Set plot labels, Show the plot
    plt.xlabel("Z-Coordinate")
    plt.ylabel("Vertex Density")
    plt.title("Vertex Density vs. Z-Coordinate")
    plt.show()

if __name__ == "__main__":
    vertex_density_plot(10, (-120, 120), 133)
