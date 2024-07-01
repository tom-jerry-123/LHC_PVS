"""
Calculates and plots vertex density
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import uproot

ROOT_PATH = "C:/Users/jerry/Documents/Phys_Summer_Research/root_files/ttbar_hist-Rel21sample.root"


def density_from_z_coord(z_coords):
    # Assume 100 reconstructed vertices on average
    MEAN_Z = 0
    STD_Z = 45
    AVE_N_VERTEX = 100
    densities = AVE_N_VERTEX * stats.norm.pdf(z_coords, loc=MEAN_Z, scale=STD_Z)
    return densities

def get_tree():
    with uproot.open(ROOT_PATH) as file:
        tree_name = "EventTree;1"
        tree = file[tree_name]

    return tree


"""
Functions for plotting vertex count
(Not used anymore)
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
    y = N * bin_size * stats.norm.pdf(x, mean, std_dev)

    # Plot the Gaussian curve
    plt.plot(x, y, label='Expected')
    plt.ylabel("Density")
    plt.xlabel("Z-Coordinate")
    plt.title("Vertex Density vs. Z-Coordinate")

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


"""
Analytical vertex density histograms
"""
def plot_hists(num_vertices):
    # Assuming 200 vertices per event, mean z = 0, std z = 45
    MEAN_Z = 0
    STD_Z = 45
    AVE_N_VERTEX = 200
    z_coord = stats.norm.rvs(loc=MEAN_Z, scale=STD_Z, size=num_vertices)
    # z_sample = stats.uniform.rvs(loc=-500, scale=1000, size=num_vertices)
    densities = AVE_N_VERTEX * stats.norm.pdf(z_coord, loc=MEAN_Z, scale=STD_Z)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Plot histograms on each subplot
    ax1.hist(z_coord, bins=50, edgecolor='black', color='blue', alpha=0.7, label='Z-Coordinate')  # Adjust bins as needed
    ax2.hist(densities, bins=50, edgecolor='black', color='salmon', alpha=0.7, label='Density')

    # Customize plot
    ax1.set_title('Z Distribution')
    ax1.set_xlabel('Z-Coordinate')
    ax2.set_title('Vertex Density Distribution')
    ax2.set_xlabel('Vertex Density')
    plt.legend()

    # Adjust layout (optional)
    plt.tight_layout()

    # Display the plot
    plt.show()



if __name__ == "__main__":
    # vertex_density_plot(10, (-120, 120), 38)
    plot_hists(10000)

