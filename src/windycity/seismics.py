#
# Ambient Noise Tomography of Canada from
# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2013JB010535
# Original data source:
# https://agupubs.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2F2013JB010535&file=Supplement1.txt
#

import math
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Cartopy imports
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def load_vs_model(filepath: Path) -> gpd.GeoDataFrame:
    """Load Vs model into a GeoDataFrame."""

    df = pd.read_csv(
        filepath, delim_whitespace=True, comment="#", header=0, engine="python"
    )

    # Header labels has longitude and latitude the wrong way around
    df = df.rename(
        columns={
            df.columns[0]: "longitude",
            df.columns[1]: "latitude",
            df.columns[2]: "depth",
            df.columns[3]: "Vs",
        }
    )

    # create a GeoDataFrame (WGS84)
    return gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326",
    )


def plot_slices(gdf: gpd.GeoDataFrame):
    """Plot Vs slices by depth using triangular interpolation."""

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
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    fig.suptitle("Shear Wave Velocity (Vs) Under Canada", fontsize=16)

    for ax, d in zip(axs.flat, plot_depths):
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
            ax.scatter(
                lon, lat, c=vs, cmap="viridis", s=20, transform=ccrs.PlateCarree()
            )
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
            cb = fig.colorbar(
                cf, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04
            )
            cb.set_label("Vs (km/s)")


def standardise(array: np.ndarray) -> np.ndarray:
    """Standardise a NumPy array to zero mean and unit variance."""
    mean = np.nanmean(array)
    std = np.nanstd(array)
    return (array - mean) / std


def pad(array: np.ndarray, target_shape: tuple[int, int] = (96, 96)) -> np.ndarray:
    """Pad a NumPy array with NaNs to reach the target shape."""
    # Compute how much padding is needed per dimension
    pad_h = target_shape[0] - array.shape[2]
    pad_w  = target_shape[1] - array.shape[3]

    pad_width = [(0, 0), (0, 0), (0, max(pad_h, 0)), (0, max(pad_w, 0))]

    # Pad with zeros
    return  np.pad(array, pad_width, mode='constant')


def to_array(gdf: gpd.GeoDataFrame, column: str) -> np.ndarray:
    """Convert a GeoDataFrame column to a NumPy array."""
    depths = np.sort(gdf["depth"].unique())
    lons = np.sort(gdf["longitude"].unique())
    lats = np.sort(gdf["latitude"].unique())

    # Preallocate with NaN
    Vs3D = np.full((len(depths), len(lats), len(lons)), np.nan)

    # Fill array with exact matches (no interpolation)
    for i, d in enumerate(depths):
        subset = gdf[gdf["depth"] == d]
        for _, row in subset.iterrows():
            lon_idx = np.where(lons == row["longitude"])[0][0]
            lat_idx = np.where(lats == row["latitude"])[0][0]
            Vs3D[i, lat_idx, lon_idx] = row["Vs"]
        Vs3D[i, :, :] = standardise(Vs3D[i, :, :])

    Vs3D = Vs3D[:, np.newaxis, :, :]  # add channel dimension
    return pad(Vs3D)


if __name__ == "__main__":
    data_file = Path() / "data" / "seismics.txt"
    vs_gdf = load_vs_model(data_file)
    plot_slices(vs_gdf)
    plt.show()
    Vs_array = to_array(vs_gdf, "Vs")
    print("Vs 3D array shape (depth, lon, lat):", Vs_array.shape)
    np.save(data_file.parent / "vs_model.npy", Vs_array)
