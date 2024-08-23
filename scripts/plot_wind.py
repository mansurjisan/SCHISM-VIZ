import numpy as np
from netCDF4 import Dataset
from netCDF4 import num2date

def create_subset_netcdf(input_file, output_file, min_lon=None, max_lon=None, min_lat=None, max_lat=None, max_steps=None, time_step=1):
    """
    Create a subset netCDF file with only the variables needed for plotting wind speed.
    
    :param input_file: Path to the input netCDF file
    :param output_file: Path to save the output subset netCDF file
    :param min_lon, max_lon, min_lat, max_lat: Spatial extent for subsetting (optional)
    :param max_steps: Maximum number of time steps to include (optional)
    :param time_step: Time step interval for subsetting (default is 1)
    """
    with Dataset(input_file, 'r') as src, Dataset(output_file, 'w') as dst:
        # Copy global attributes
        dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})

        # Read coordinates
        lon = src.variables['SCHISM_hgrid_node_x'][:]
        lat = src.variables['SCHISM_hgrid_node_y'][:]

        # Create spatial mask
        if all([min_lon, max_lon, min_lat, max_lat]):
            mask = (lon >= min_lon) & (lon <= max_lon) & (lat >= min_lat) & (lat <= max_lat)
        else:
            mask = np.ones_like(lon, dtype=bool)

        # Determine time subset
        times = src.variables['time'][:]
        if max_steps is not None:
            time_indices = slice(0, min(max_steps, len(times)), time_step)
        else:
            time_indices = slice(None, None, time_step)

        # Create dimensions
        dst.createDimension('node', np.sum(mask))
        dst.createDimension('time', len(times[time_indices]))
        dst.createDimension('vec', 2)  # Assuming wind speed has x and y components

        # Create and populate variables
        # Time
        time_var = dst.createVariable('time', src.variables['time'].datatype, ('time',))
        time_var.setncatts({a: src.variables['time'].getncattr(a) for a in src.variables['time'].ncattrs()})
        time_var[:] = times[time_indices]

        # Longitude
        lon_var = dst.createVariable('SCHISM_hgrid_node_x', src.variables['SCHISM_hgrid_node_x'].datatype, ('node',))
        lon_var.setncatts({a: src.variables['SCHISM_hgrid_node_x'].getncattr(a) for a in src.variables['SCHISM_hgrid_node_x'].ncattrs()})
        lon_var[:] = lon[mask]

        # Latitude
        lat_var = dst.createVariable('SCHISM_hgrid_node_y', src.variables['SCHISM_hgrid_node_y'].datatype, ('node',))
        lat_var.setncatts({a: src.variables['SCHISM_hgrid_node_y'].getncattr(a) for a in src.variables['SCHISM_hgrid_node_y'].ncattrs()})
        lat_var[:] = lat[mask]

        # Wind speed
        wind_var = dst.createVariable('wind_speed', src.variables['wind_speed'].datatype, ('time', 'node', 'vec'))
        wind_var.setncatts({a: src.variables['wind_speed'].getncattr(a) for a in src.variables['wind_speed'].ncattrs()})
        wind_var[:] = src.variables['wind_speed'][time_indices][:, mask, :]

    print(f"Subset netCDF file created: {output_file}")


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#from geocat.viz import cmaps as gvcmaps
from geocat.viz import util as gvutil
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
from netCDF4 import Dataset
import geocat.viz as gv
import os
from tqdm import tqdm
import time
import datetime as dt
from netCDF4 import num2date
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

def plot_wind_speed_sequence(netcdf_file, output_dir, min_lon=None, max_lon=None, min_lat=None, max_lat=None, plot_vectors=True, vmin=None, vmax=None):
    """
    Create a sequence of wind speed plots from the SCHISM output netCDF file.

    :param netcdf_file: Path to the SCHISM output netCDF file
    :param output_dir: Directory to save output plots
    :param min_lon, max_lon, min_lat, max_lat: Plot extent (optional)
    :param plot_vectors: Whether to plot wind vectors (default is True)
    :param vmin, vmax: Minimum and maximum values for wind speed colorbar (optional)
    """
    # Open the netCDF file
    nc = Dataset(netcdf_file, 'r')

    # Read data
    lon = nc.variables['SCHISM_hgrid_node_x'][:]
    lat = nc.variables['SCHISM_hgrid_node_y'][:]
    times = nc.variables['time'][:]
    time_units = nc.variables['time'].units
    wind_speed = nc.variables['wind_speed'][:]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set up the plot
    proj = ccrs.PlateCarree()

    # Loop through time steps
    for i, time_value in enumerate(tqdm(times, desc="Creating plots")):

        shapefile_path = '/home/mjisan/.local/share/cartopy/shapefiles/natural_earth/physical/ne_10m_admin_1_states_provinces_lakes.shp'

        # Create a feature from the shapefile
        states_provinces = ShapelyFeature(Reader(shapefile_path).geometries(),
                                          ccrs.PlateCarree(),
                                          edgecolor='black', facecolor='none')
        
        fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': proj})
        plt.xlabel('')
        plt.ylabel('')
        # Convert time to datetime object
        datetime_obj = num2date(time_value, units=time_units)

        # Format datetime object to string in 'YYYY-MM-DD HH:MM' format
        time_str = datetime_obj.strftime('%Y-%m-%d %H:%M')
        print(time_str)


        # Set extent if provided
        if all([min_lon, max_lon, min_lat, max_lat]):
            ax.set_extent([min_lon, max_lon, min_lat, max_lat])
        else:
            ax.set_global()

        ax.add_feature(states_provinces)
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.LAND, facecolor='white')

#        ax.add_feature(cfeature.LAND, facecolor='white')
#        ax.add_feature(cfeature.COASTLINE)
#        ax.add_feature(cfeature.BORDERS)
#        ax.add_feature(cfeature.LAND)
#        ax.coastlines(resolution='10m', color='black', linewidth=0.25)
#        ax.add_feature(cfeature.STATES, linewidth=0.25)
            
#        ax.add_feature(cfeature.COASTLINE)
#        ax.add_feature(cfeature.BORDERS)
#        ax.add_feature(cfeature.LAND, facecolor="lightgrey", zorder=0)
            
        ## Add coastline and other features
        #ax.add_feature(cfeature.LAND, facecolor="lightgrey", zorder=0)
        #ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.5, zorder=1)
        #ax.add_feature(cfeature.BORDERS, edgecolor='gray', linestyle=':', linewidth=0.5, zorder=1)

        # Calculate wind magnitude
        wind_magnitude = np.sqrt(wind_speed[i, :, 0]**2 + wind_speed[i, :, 1]**2)

        # Calculate min and max wind speed
        min_wind = np.min(wind_magnitude)
        max_wind = np.max(wind_magnitude)


        # Plot wind speed magnitude
        levels = np.linspace(vmin or np.min(wind_magnitude), vmax or np.max(wind_magnitude), 25)
        levels = np.round(levels, 0)  # Rounds to the nearest integer
        cmap = plt.get_cmap('jet')
        cs = ax.tricontourf(lon, lat, wind_magnitude, levels=levels, cmap=cmap, transform=proj, zorder=1)

        # Add vectors if requested
        if plot_vectors:
            quiver_stride = max(1, len(lon) // 1000)  # Adjust stride based on data size
            Q = ax.quiver(lon[::quiver_stride], lat[::quiver_stride],
                          wind_speed[i, ::quiver_stride, 0], wind_speed[i, ::quiver_stride, 1],
                          color='white', scale=50, width=0.002, transform=proj, zorder=4)

            # Add quiver key
            ax.quiverkey(Q, 0.95, 0.95, 5, '5 m/s', labelpos='N', coordinates='figure')

        # Customize tick marks and labels
        gv.add_lat_lon_ticklabels(ax)
        gvutil.set_axes_limits_and_ticks(ax,
                                         xticks=[-86,  -84,  -82, -80, -78],
                                         yticks=[22, 26, 30, 34])

        gvutil.add_major_minor_ticks(ax,
                                     x_minor_per_major=2,
                                     y_minor_per_major=2,
                                     labelsize=10)
        
        gvutil.set_titles_and_labels(ax,
                                     maintitle="Surface Wind Speed",
                                     maintitlefontsize=10,
                                     lefttitle=f"Time: {time_str}",
                                     lefttitlefontsize=10,
                                     righttitle="Surface Wind Speed (m/s)",
                                     righttitlefontsize=10,
                                     xlabel="Longitude",
                                     ylabel="Latitude")
    

        plt.tick_params(
            axis='x',          
            which='both',     
            bottom=True,      
            top=False,        
            labelbottom=True) 

        plt.tick_params(
            axis='y',         
            which='both',    
            left=True,      
            right=False,        
            labelbottom=True) 

        ax.coastlines(linewidths=1.5)
        ax.xaxis.set_tick_params(labelsize=10)
        ax.yaxis.set_tick_params(labelsize=10)
        
        ax.add_feature(cfeature.LAND, facecolor='white')
        ax.tick_params(labelsize=12) 
        mpl.rcParams['xtick.labelsize'] = 12
        mpl.rcParams['ytick.labelsize'] = 12
        
#        gv.set_axes_limits_and_ticks(ax,
#                                     xlim=(min_lon, max_lon),
#                                     ylim=(min_lat, max_lat),
#                                     xticks=np.arange(min_lon, max_lon+1, 5),
#                                     yticks=np.arange(min_lat, max_lat+1, 5))
#        gv.add_major_minor_ticks(ax, x_minor_per_major=4, y_minor_per_major=4, labelsize=10)

        # Remove degree symbol from tick labels
#        ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol=''))
#        ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol=''))

        # Set titles and labels
        gv.set_titles_and_labels(ax,
                                 maintitle=f'Hurricane Ian Surface Wind Speed',
                                 maintitlefontsize=12,
                                 lefttitle=f"{time_str}",
                                 lefttitlefontsize=8,
#                                 righttitle="m/s",
                                 righttitle=f"min | max = {min_wind:.2f} | {max_wind:.2f}",
                                 righttitlefontsize=8,
                                 xlabel="",
                                 ylabel="")

        # Add colorbar
#        cbar = plt.colorbar(cs, ax=ax, orientation='vertical', pad=0.02, aspect=30)
        cbar = plt.colorbar(cs, ax=ax, orientation='vertical', pad=0.02, aspect=20, fraction=0.046)
        cbar.ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

        cbar.set_label('Wind Speed (m/s)', fontsize=12)

        # Save the plot
        output_file = os.path.join(output_dir, f'wind_speed_plot_{i:04d}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        # Print timing information for this plot
        print(f"Plot {i+1}/{len(times)} generated.")

    # Close the netCDF file
    nc.close()

    print(f"\nAll plots saved in {output_dir}")

# Usage example
if __name__ == "__main__":
    input_file = "schout_wind_3.nc"
    subset_file = "schout_wind_3_subset.nc"
    create_subset_netcdf(input_file, subset_file,
                         min_lon=-86, max_lon=-76, min_lat=22, max_lat=34,
                         max_steps=120, time_step=12)
#
    netcdf_file = "schout_wind_3_subset.nc"
    output_dir = "wind_speed_plots_3"
    plot_wind_speed_sequence(netcdf_file, output_dir,
                             min_lon=-86, max_lon=-78, min_lat=22, max_lat=34,
                             plot_vectors=False, vmin=0, vmax=40)
