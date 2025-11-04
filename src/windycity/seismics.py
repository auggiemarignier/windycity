#
# Ambient Noise Tomography of Canada from
# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2013JB010535
#

import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Cartopy imports
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# 1) load file
filepath = "data/seismics.txt"
# read with whitespace delimiting; skip empty lines
df = pd.read_csv(
    filepath, delim_whitespace=True, comment="#", header=0, engine="python"
)

# If header labels truly swapped, file's header says "lat lon ..." but they are reversed.
# Create correctly-named columns: latitude (originally lon) and longitude (originally lat).
df = df.rename(
    columns={
        df.columns[0]: "longitude",
        df.columns[1]: "latitude",
        df.columns[2]: "depth",
        df.columns[3]: "Vs",
    }
)

# create a GeoDataFrame (WGS84)
gdf = gpd.GeoDataFrame(
    df,
    geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
    crs="EPSG:4326",
)

# 2) prepare depth slices
depths = sorted(gdf["depth"].unique())
# choose up to 6 depths to plot (if many), spaced evenly through list
max_plots = 6
if len(depths) > max_plots:
    idx = np.linspace(0, len(depths) - 1, max_plots, dtype=int)
    plot_depths = [depths[i] for i in idx]
else:
    plot_depths = depths

# set up figure with one subplot per depth (arranged in grid)
n = len(plot_depths)
ncols = min(3, n)
nrows = math.ceil(n / ncols)
fig = plt.figure(figsize=(5 * ncols, 4 * nrows))
plt.suptitle("Vs slices by depth (m/s) — triangular interpolation", fontsize=16)

for i, d in enumerate(plot_depths, start=1):
    ax = fig.add_subplot(nrows, ncols, i, projection=ccrs.PlateCarree())
    subset = gdf[gdf["depth"] == d]
    if subset.empty:
        ax.set_title(f"Depth {d} (no data)")
        continue

    lon = subset["longitude"].values
    lat = subset["latitude"].values
    vs = subset["Vs"].values.astype(float)

    # Create a triangulation for tricontourf interpolation
    try:
        triang = tri.Triangulation(lon, lat)
        # set levels
        vmin, vmax = np.nanmin(vs), np.nanmax(vs)
        levels = np.linspace(vmin, vmax, 12)
        cf = ax.tricontourf(
            triang, vs, levels=levels, transform=ccrs.PlateCarree(), cmap="viridis"
        )
    except Exception as e:
        # fallback to simple scatter if triangulation fails
        ax.scatter(lon, lat, c=vs, cmap="viridis", s=20, transform=ccrs.PlateCarree())
        cf = None

    # plot coastlines and gridlines for context
    ax.coastlines(resolution="50m", linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4)
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(f"Depth = {d}")

    # set extent slightly larger than data bounds for this subset
    pad = 1.0
    lon_min, lon_max = lon.min() - pad, lon.max() + pad
    lat_min, lat_max = lat.min() - pad, lat.max() + pad
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    if cf is not None:
        cb = fig.colorbar(cf, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
        cb.set_label("Vs")

# save figure to disk for download/viewing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
