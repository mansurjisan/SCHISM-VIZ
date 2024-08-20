import numpy as numpy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

print(f"NumPy version: {numpy.__version__}")

def read_gr3_file(filename):
    try:
        with open(filename, 'r') as f:
            _ = f.readline()  # Discard first line (comment)
            ne, np = map(int, f.readline().split())

            nodes = numpy.empty((np, 3))
            for i in range(np):
                nodes[i] = list(map(float, f.readline().split()[1:4]))

            elements = []
            for _ in range(ne):
                elements.append(list(map(lambda x: int(x) - 1, f.readline().split()[2:])))

        print(f"Number of nodes: {np}")
        print(f"Number of elements: {ne}")
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
    fig = plt.figure(figsize=(20, 20))
    
    # Use PlateCarree projection
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Add coastlines and borders
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, edgecolor='gray', linestyle=':', linewidth=0.5)

    # Add grid lines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Create line segments for all elements
    segs = []
    for element in elements:
        points = nodes[element, :2]
        segs.append(numpy.concatenate([points, points[[0]]]))  # Close the polygon

    print("Creating line collection...")
    line_segments = LineCollection(segs, linewidths=0.1, colors='blue', alpha=0.5)

    print("Adding line collection to plot...")
    ax.add_collection(line_segments)

    # Set map extent based on node coordinates
    margin = 0.5  # degrees
    ax.set_extent([
        nodes[:, 0].min() - margin,
        nodes[:, 0].max() + margin,
        nodes[:, 1].min() - margin,
        nodes[:, 1].max() + margin
    ])

    # Add a color bar to represent depth
    scatter = ax.scatter(nodes[:, 0], nodes[:, 1], c=nodes[:, 2], cmap='viridis', 
                         s=1, alpha=0.5, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Depth (m)', rotation=270, labelpad=15)

    # Set labels and title
    ax.set_title('SCHISM Grid Plot with Coastlines', fontsize=16, pad=20)

    print(f"Saving plot to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Plot saved successfully.")

# Usage
if __name__ == "__main__":
    filename = 'hgrid.gr3'
    output_file = 'schism_grid_plot_with_coastlines.png'

    print(f"Reading file: {filename}")
    nodes, elements = read_gr3_file(filename)
    if nodes is not None and elements is not None:
        plot_gr3(nodes, elements, output_file)
    else:
        print("Failed to read the .gr3 file. Please check the file path and format.")
