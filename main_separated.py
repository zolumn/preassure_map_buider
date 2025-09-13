#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import os, argparse, warnings, yaml
from pathlib import Path  # Add this import
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

from scipy.interpolate import griddata
from pykrige.ok import OrdinaryKriging
from rasterio.transform import from_origin
import rasterio

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------
# CONFIG
# -------------------------

def read_surfer_colormap(clr_path):
    """
    Read Surfer .clr file and convert it to matplotlib colormap
    
    Parameters:
    -----------
    clr_path : str or Path
        Path to Surfer .clr file
        
    Returns:
    --------
    matplotlib.colors.LinearSegmentedColormap
    """
    colors = []
    
    with open(clr_path, 'r') as f:
        # Skip header line
        header = f.readline()
        if not header.startswith('ColorMap'):
            raise ValueError(f"Invalid colormap file format: {clr_path}")
            
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Parse color values
            parts = line.strip().split()
            if len(parts) >= 5:  # position R G B A format
                try:
                    r = int(parts[1]) / 255
                    g = int(parts[2]) / 255
                    b = int(parts[3]) / 255
                    colors.append((r, g, b))
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping invalid line in colormap file: {line.strip()}")
                    continue
    
    if not colors:
        raise ValueError(f"No valid colors found in {clr_path}")
    
    # Create colormap with 100 bins for smooth interpolation
    return LinearSegmentedColormap.from_list('geology', colors, N=100)

def load_cfg(path):
    """Load and process configuration"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Try to load geology colormap
    clr_path = Path(Path(path).parent / "geology.clr") # Look for clr file in same dir as config
    if clr_path.exists():
        try:
            # Create and register the colormap
            geology_cmap = read_surfer_colormap(clr_path)
            plt.colormaps.register(geology_cmap)
            
            # Set as default in config
            if "plot" not in cfg:
                cfg["plot"] = {}
            cfg["plot"]["cmap"] = 'geology'
            
            print(f"[DEBUG] Successfully loaded colormap from {clr_path}")
        except Exception as e:
            print(f"[WARNING] Failed to load geology colormap: {e}")
            # Set fallback colormap
            if "plot" not in cfg:
                cfg["plot"] = {}
            cfg["plot"]["cmap"] = "gist_earth_r"
    
    return cfg

# -------------------------
# DATA READERS
# -------------------------

def _lower_map(cols):
    return {c.lower(): c for c in cols}

def _autodetect(colnames, candidates):
    """Find first existing (case-insensitive) column among candidates list."""
    low = _lower_map(colnames)
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None

def read_static(path, cfg):
    """
    Reads static coordinates and optional TVD/Pbp from static CSV.
    Required (via config): name, x, y.
    Optional (auto-detected if not specified in config): tvd, pbp, bt_x, bt_y, md.
    """
    sep = cfg.get("csv_sep", ",")
    scol = cfg.get("static_columns", {})
    df = pd.read_csv(path, delimiter=sep)
    df.columns = df.columns.str.strip()
    low = _lower_map(df.columns)

    # Required by config
    for req in ("name", "x", "y", "bt_x", "bt_y"):
        if req not in scol or scol[req].lower() not in low:
            raise ValueError(f"Static file is missing column specified in config: {scol.get(req)}")
    
    name_col = low[scol["name"].lower()]
    xcol     = low[scol["x"].lower()]
    ycol     = low[scol["y"].lower()]
    bt_xcol  = low[scol["bt_x"].lower()]
    bt_ycol  = low[scol["bt_y"].lower()]

    # Optional columns - try config mapping first; else autodetect by name
    tvd_col = None
    pbp_col = None
    if "tvd" in scol and scol["tvd"].lower() in low:
        tvd_col = low[scol["tvd"].lower()]
    else:
        tvd_col = _autodetect(df.columns, ["TVD","tvd"])

    if "pbp" in scol and scol["pbp"].lower() in low:
        pbp_col = low[scol["pbp"].lower()]
    else:
        pbp_col = _autodetect(df.columns, ["Pbp","pbp","PBP"])

    keep = [name_col, xcol, ycol, bt_xcol, bt_ycol]
    if tvd_col: keep.append(tvd_col)
    if pbp_col: keep.append(pbp_col)

    out = df[keep].copy()
    rename_map = {
        name_col: "name", 
        xcol: "x", 
        ycol: "y",
        bt_xcol: "bt_x",
        bt_ycol: "bt_y"
    }
    if tvd_col: rename_map[tvd_col] = "tvd"
    if pbp_col: rename_map[pbp_col] = "pbp"
    out.rename(columns=rename_map, inplace=True)

    out["name"] = out["name"].astype(str)
    out = out.dropna(subset=["bt_x","bt_y"]).drop_duplicates(subset=["name"], keep="first")
    return out, "bt_x", "bt_y", "name"

def read_dynamic(path, cfg):
    """
    Reads dynamic measurements: Name + (Pres and/or BHP).
    Returns dataframe with columns: name, pres?, bhp?
    """
    sep = cfg.get("csv_sep", ",")
    dcol = cfg.get("dynamic_columns", {})
    df = pd.read_csv(path, delimiter=sep)
    df.columns = df.columns.str.strip()
    low = _lower_map(df.columns)

    if "name" not in dcol or dcol["name"].lower() not in low:
        raise ValueError(f"Dynamic file is missing name column specified in config: {dcol.get('name')}")
    name_col = low[dcol["name"].lower()]

    pres_col = dcol.get("pres")
    bhp_col  = dcol.get("bhp")
    pres_col = low[pres_col.lower()] if (pres_col and pres_col.lower() in low) else _autodetect(df.columns, ["Pres","pres","PRES"])
    bhp_col  = low[bhp_col.lower()]  if (bhp_col  and bhp_col.lower()  in low) else _autodetect(df.columns, ["BHP","bhp","BhP"])

    if not pres_col and not bhp_col:
        raise ValueError("Dynamic file must contain at least one of Pres/BHP columns.")

    keep = [name_col]
    if pres_col: keep.append(pres_col)
    if bhp_col:  keep.append(bhp_col)

    out = df[keep].copy()
    ren = {name_col: "name"}
    if pres_col: ren[pres_col] = "pres"
    if bhp_col:  ren[bhp_col]  = "bhp"
    out.rename(columns=ren, inplace=True)
    out["name"] = out["name"].astype(str)

    # Drop rows where BHP is NaN
    if "bhp" in out.columns:
        out = out.dropna(subset=["bhp"])
    # Drop rows where Pres is NaN
    if "pres" in out.columns:
        out = out.dropna(subset=["pres"])   

    # Aggregate duplicates by mean for numeric columns
    num_cols = [c for c in out.columns if c != "name"]
    out = (out.dropna(subset=num_cols, how="all")
              .groupby("name", as_index=False)[num_cols].mean())
    return out

# -------------------------
# GRID INTERPOLATION
# -------------------------

def build_grid(df, xcol, ycol, nx, ny, margin):
    x = df[xcol].values; y = df[ycol].values
    xmin, xmax = x.min() - margin, x.max() + margin
    ymin, ymax = y.min() - margin, y.max() + margin
    gx = np.linspace(xmin, xmax, nx)
    gy = np.linspace(ymin, ymax, ny)
    GX, GY = np.meshgrid(gx, gy)
    print(f"[DEBUG] Grid bounds: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
    return GX, GY, gx, gy, (xmin, xmax, ymin, ymax)

def interpolate(df, xcol, ycol, zcol, GX, GY, method='kriging'):
    # Remove rows with NaN in target column
    df_clean = df.dropna(subset=[zcol])
    
    if len(df_clean) < 4:
        raise ValueError(f"Too few valid points ({len(df_clean)}) for interpolation")
    
    # Extract coordinates and values
    x = df_clean[xcol].values
    y = df_clean[ycol].values
    z = df_clean[zcol].values
    
    print(f"[DEBUG] Interpolation input:")
    print(f"Points count: {len(z)}")
    print(f"Value range: {z.min():.2f} to {z.max():.2f}")
    
    try:
        if method.lower() == 'kriging':
            # Get min/max for coordinates to set variogram parameters
            x_range = x.max() - x.min()
            y_range = y.max() - y.min()
            avg_range = (x_range + y_range) / 2
            
            # Calculate variogram parameters
            sill = np.var(z)  # Use variance of data as sill
            variogram_parameters = {
                'sill': sill,
                'range': avg_range / 3,  # Use 1/3 of average range
                'nugget': 0.0  # Start with no nugget effect
            }
            
            # Set up the kriging model with explicit variogram parameters
            ok = OrdinaryKriging(
                x, y, z,
                variogram_model='spherical',
                variogram_parameters=variogram_parameters,
                nlags=10,
                weight=True,
                verbose=False,
                enable_plotting=False
            )
            
            # Get grid points for kriging
            grid_x = np.unique(GX[0, :])
            grid_y = np.unique(GY[:, 0])
            
            # Perform kriging interpolation
            Z, ss = ok.execute(
                'grid', 
                grid_x,
                grid_y
            )
            
            # Verify results
            if not np.isnan(Z).all():
                print("[DEBUG] Kriging validation:")
                print(f"Input range: {z.min():.2f} to {z.max():.2f}")
                print(f"Output range: {np.nanmin(Z):.2f} to {np.nanmax(Z):.2f}")
                print(f"Variogram parameters used:")
                print(f"  - Sill: {variogram_parameters['sill']:.2f}")
                print(f"  - Range: {variogram_parameters['range']:.2f}")
                print(f"  - Nugget: {variogram_parameters['nugget']:.2f}")
                
                # If kriging result is constant, fall back to linear
                if np.allclose(Z, Z[0,0], rtol=1e-5):
                    print("[DEBUG] Kriging produced constant value, falling back to linear")
                    Z = griddata((x, y), z, (GX, GY), method='linear')
                    return Z, "linear (kriging fallback)"
                
                return Z, "ordinary kriging"
            
            raise ValueError("Kriging produced invalid results")
            
        elif method.lower() in ['linear', 'cubic', '3-nearest']:
            # For non-kriging methods, use scipy's griddata
            actual_method = 'nearest' if method.lower() == '3-nearest' else method.lower()
            Z = griddata((x, y), z, (GX, GY), method=actual_method)
            
            if not np.isnan(Z).all():
                return Z, f"griddata {actual_method}"
            
            raise ValueError(f"{method} interpolation failed")
            
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
            
    except Exception as e:
        print(f"[DEBUG] {method} interpolation failed: {str(e)}")
        print("[DEBUG] Falling back to 3-nearest neighbor interpolation")
        
        # Perform nearest neighbor interpolation as fallback
        Z = griddata(
            (x, y), z,
            (GX, GY),
            method='nearest'
        )
        
        return Z, "nearest neighbor (fallback)"
    
    raise ValueError("All interpolation methods failed")

# -------------------------
# PLOT AND EXPORT
# -------------------------

def save_png(df, xcol, ycol, zcol, GX, GY, Z, cfg, out_png, z_label=None, title_suffix=""):
    if np.isnan(Z).all():
        raise ValueError("Interpolated Z array contains only NaN values. Check input data and interpolation settings.")
    
    fig, ax = plt.subplots(figsize=tuple(cfg["plot"]["figsize"]))
    ax.set_axis_off()

    # Calculate data range from both original data and interpolated values
    orig_zmin = float(df[zcol].min())
    orig_zmax = float(df[zcol].max())
    int_zmin = float(np.nanmin(Z))
    int_zmax = float(np.nanmax(Z))
    
    # Use the wider range
    zmin = min(orig_zmin, int_zmin)
    zmax = max(orig_zmax, int_zmax)
    
    print(f"[DEBUG] Value ranges:")
    print(f"Original data: {orig_zmin:.2f} to {orig_zmax:.2f}")
    print(f"Interpolated: {int_zmin:.2f} to {int_zmax:.2f}")
    print(f"Final range: {zmin:.2f} to {zmax:.2f}")
    
    # Create evenly spaced levels
    levels = np.linspace(zmin, zmax, 21)  # 21 boundaries create 20 intervals
    
    print(f"[DEBUG] Level boundaries: {levels}")

    # Use the same levels for both contourf and contour
    cmap = cfg["plot"].get("cmap", "gist_earth_r")
    cs = ax.contourf(GX, GY, Z, levels=levels, cmap=cmap, extend='both')
    c = ax.contour(GX, GY, Z, colors="k", linewidths=0.4, levels=levels[::2])
    
    if cfg["plot"].get("draw_contour_labels", True):
        ax.clabel(c, inline=True, fontsize=6, fmt="%.1f", levels=levels[::2])  # Reduced fontsize from 8 to 6

    # Plot wells
    ax.scatter(df[xcol], df[ycol], s=26, marker="^", facecolors="white", edgecolors="k", linewidths=0.5)

    if cfg["plot"].get("show_labels", True):
        name_col = cfg.get("csv_columns", {}).get("name", "name")
        dx = cfg["plot"].get("label_dx", 3)
        dy = cfg["plot"].get("label_dy", 3)
        for _, r in df.iterrows():
            ax.annotate(
                str(r[name_col]),
                (r[xcol], r[ycol]),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=cfg["plot"].get("label_fontsize", 7),
                ha="left", va="bottom",
                color='black',  # Added explicit black color for well labels
                #path_effects=[pe.withStroke(linewidth=1.5), foreground="white"]
            )

    ttl = cfg["plot"].get("title", "Pressure map")
    if title_suffix:
        ttl = f"{ttl} — {title_suffix}"
    ax.set_title(ttl, pad=10)

    cbar_label = z_label if z_label else cfg.get("z_label", zcol)
    fig.colorbar(cs, ax=ax, label=cbar_label)

    fig.tight_layout()
    fig.savefig(out_png, dpi=cfg["plot"].get("dpi", 300))
    plt.close(fig)

def save_geotiff(Z, bounds, nx, ny, epsg, nodata, out_tif):
    xmin, xmax, ymin, ymax = bounds
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    transform = from_origin(xmin, ymax, dx, dy)
    Z_out = np.nan_to_num(Z, nan=nodata)
    with rasterio.open(
        out_tif, "w",
        driver="GTiff",
        height=Z_out.shape[0],
        width=Z_out.shape[1],
        count=1, dtype=Z_out.dtype,
        crs=f"EPSG:{epsg}",
        transform=transform, compress="lzw", nodata=nodata
    ) as dst:
        dst.write(Z_out, 1)

# -------------------------
# DERIVED COLUMNS
# -------------------------

def compute_derived(df):
    """
    Adds:
      pres_ref_2434 = Pres + (2434 - TVD)*950*9.81/100000
      diffbhp = BHP - Pbp
    """
    # Hydrostatic correction constants
    REF_DEPTH = 2434.0  # m
    RHO       = 950.0   # kg/m3
    G         = 9.81    # m/s2
    DIVISOR   = 100000.0  # Pa -> bar

    if "pres" in df.columns and "tvd" in df.columns:
        df["pres_ref_2434"] = df["pres"] + (REF_DEPTH - df["tvd"]) * RHO * G / DIVISOR

    if "bhp" in df.columns and "pbp" in df.columns:
        df["diffbhp"] = df["bhp"] - df["pbp"]
    return df

# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Pressure map builder (static+dynamic split, CLI z-column).")
    p.add_argument("--config", "-c", default=os.environ.get("CONFIG", "app/config/config_separated.yaml"),
                   help="Path to YAML config (defaults to CONFIG env or ./app/config/config_separated.yaml)")
    p.add_argument("--z-column", help="Column to plot (e.g., pres, bhp, pres_ref_2434, diffbhp). If omitted, auto-select.")
    p.add_argument("--z-label", help="Colorbar label override.")
    return p.parse_args()

# -------------------------
# MAIN
# -------------------------

def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    # Read static & dynamic
    static_df, xcol, ycol, namecol = read_static(cfg["input_static"], cfg)
    dynamic_df = read_dynamic(cfg["input_dynamic"], cfg)

    # Merge on name
    df = static_df.merge(dynamic_df, on="name", how="inner")
    print("[DEBUG] Merged DataFrame:")
    print(df.head())
    print(df.info())

    # Compute derived metrics
    df = compute_derived(df)

    print("[DEBUG] After computing additional metrix:")
    print(df.head())
    df.to_csv(f'{pathlib.Path(cfg["output_dir"])}/debug_merged_data.csv')
    
    # Choose z column
    z_column = args.z_column
    if not z_column:
        for cand in ("bhp", "pres", "pres_ref_2434", "diffbhp"):
            if cand in df.columns:
                z_column = cand
                break
    if not z_column:
        numeric_cols = [c for c in df.columns if c not in ["name", xcol, ycol, "tvd", "pbp"] and np.issubdtype(df[c].dtype, np.number)]
        if not numeric_cols:
            raise ValueError("No numeric column available to plot.")
        z_column = numeric_cols[0]

    print(f"[DEBUG] Selected z_column: {z_column}")
    print(df[[xcol, ycol, z_column]].dropna())

    # Build grid and interpolate using bottom-hole coordinates
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    margin = float(cfg.get("bounds_margin_units", 0.0))
    GX, GY, gx, gy, bounds = build_grid(df, xcol, ycol, nx, ny, margin)
    Z, info = interpolate(df, xcol, ycol, z_column, GX, GY, method=cfg["interpolation"]["method"])
    print(f"[INFO] Interpolation: {info}, points={len(df)}, grid=({ny},{nx}), z_column={z_column}")
    print(f"[DEBUG] Interpolated Z array (shape={Z.shape}):")
    print(Z)

    # Export
    outdir = pathlib.Path(cfg["output_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    stem = pathlib.Path(cfg["input_dynamic"]).stem
    png_path = outdir / f"{stem}_pressure_map_{z_column}.png"
    save_png(df, xcol, ycol, z_column, GX, GY, Z, cfg, png_path, z_label=args.z_label, title_suffix=z_column)
    print(f"[OK] PNG saved: {png_path}")

    if cfg["export"].get("geotiff", True):
        tif_path = outdir / f"{stem}_pressure_map_{z_column}.tif"
        save_geotiff(Z, bounds, nx, ny, cfg["crs_epsg"], cfg["export"]["geotiff_nodata"], tif_path)
        print(f"[OK] GeoTIFF saved: {tif_path}")

if __name__ == "__main__":
    main()
