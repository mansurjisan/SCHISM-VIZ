import numpy as numpy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os

print(f"NumPy version: {numpy.__version__}")
print(f"NumPy type: {type(numpy)}")

def read_gr3_file(filename):
    try:
        with open(filename, 'r') as f:
            # Read header
            _ = f.readline()  # Discard first line (comment)
            ne, np = map(int, f.readline().split())

            # Read node coordinates
            nodes = numpy.empty((np, 3))
            for i in range(np):
                nodes[i] = list(map(float, f.readline().split()[1:4]))

            # Read element connectivity
            elements = []
            for _ in range(ne):
                elements.append(list(map(lambda x: int(x) - 1, f.readline().split()[2:])))

        print(f"Number of nodes: {np}")
        print(f"Number of elements: {ne}")
        print(f"Sample element: {elements[0]}")
        
        return nodes, elements
    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def plot_gr3(nodes, elements, output_file):
    if nodes is None or elements is None:
        print("Cannot plot: invalid data")
        return

    print("Preparing plot...")
    fig, ax = plt.subplots(figsize=(20, 20))  # Increased figure size

    # Create line segments for all elements
    segs = []
    for element in elements:
        points = nodes[element, :2]
        segs.append(numpy.concatenate([points, points[[0]]]))  # Close the polygon

    print("Creating line collection...")
    line_segments = LineCollection(segs, linewidths=0.1, colors='b')

    print("Adding line collection to plot...")
    ax.add_collection(line_segments)

    # Set axis limits
    ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
    ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())

    # Set labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('SCHISM Grid Plot')

    # Equal aspect ratio
    ax.set_aspect('equal', 'box')

    print(f"Saving plot to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Plot saved successfully.")

# Usage
if __name__ == "__main__":
    filename = 'hgrid.gr3'
    output_file = 'schism_grid_plot.png'

    print(f"Reading file: {filename}")
    nodes, elements = read_gr3_file(filename)
    if nodes is not None and elements is not None:
        plot_gr3(nodes, elements, output_file)
    else:
        print("Failed to read the .gr3 file. Please check the file path and format.")
