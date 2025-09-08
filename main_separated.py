
import os, sys, pathlib, warnings, yaml
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from scipy.interpolate import griddata
from pykrige.ok import OrdinaryKriging
from rasterio.transform import from_origin
import rasterio

warnings.filterwarnings("ignore", category=UserWarning)

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _normalize_cols(df):
    # keep original header names but add a lowercase alias map
    df.columns = df.columns.str.strip()
    lower_map = {c: c.lower() for c in df.columns}
    df.columns = [c for c in df.columns]  # keep original
    return lower_map

def read_static(path, cfg):
    sep = cfg.get("csv_sep", ",")
    cols = cfg["static_columns"]
    df = pd.read_csv(path, delimiter=sep)
    df.columns = df.columns.str.strip()
    # work case-insensitively
    lower = {c.lower(): c for c in df.columns}
    required = [cols["name"], cols["x"], cols["y"]]
    missing = [c for c in required if c.lower() not in lower]
    if missing:
        raise ValueError(f"Static file is missing columns: {missing}")
    name_col = lower[cols["name"].lower()]
    xcol = lower[cols["x"].lower()]
    ycol = lower[cols["y"].lower()]
    # Optional columns
    opt = {}
    for k in ["bt_x","bt_y","md","tvd_pbp"]:
        v = cols.get(k)
        if v and v.lower() in lower:
            opt[k] = lower[v.lower()]
    out = df[[name_col, xcol, ycol] + list(opt.values())].copy()
    out.rename(columns={name_col:"name", xcol:"x", ycol:"y"}, inplace=True)
    out["name"] = out["name"].astype(str)
    # drop duplicates by name keeping first; coordinates assumed static
    out = out.dropna(subset=["x","y"]).drop_duplicates(subset=["name"], keep="first")
    return out, "x", "y", "name", opt

def read_dynamic(path, cfg):
    sep = cfg.get("csv_sep", ",")
    cols = cfg["dynamic_columns"]
    df = pd.read_csv(path, delimiter=sep)
    df.columns = df.columns.str.strip()
    lower = {c.lower(): c for c in df.columns}
    # Required: name and at least one of Pres, BHP
    name_key = cols["name"]
    if name_key.lower() not in lower:
        raise ValueError(f"Dynamic file is missing name column: {name_key}")
    name_col = lower[name_key.lower()]
    z_candidates = []
    if cols.get("pres") and cols["pres"].lower() in lower:
        z_candidates.append(("pres", lower[cols["pres"].lower()]))
    if cols.get("bhp") and cols["bhp"].lower() in lower:
        z_candidates.append(("bhp", lower[cols["bhp"].lower()]))
    if not z_candidates:
        raise ValueError("Dynamic file must contain at least one of Pres/BHP columns.")
    # choose z by preference order
    pref = [z.strip().lower() for z in cfg.get("z_preference", ["bhp","pres"])]
    use = None
    for key, col in z_candidates:
        if key in pref and use is None:
            use = (key, col)
    if use is None:
        # fallback to first available
        use = z_candidates[0]
    zkey, zcol = use
    out = df[[name_col, zcol]].copy()
    out.rename(columns={name_col:"name", zcol:"z"}, inplace=True)
    out["name"] = out["name"].astype(str)
    # aggregate duplicates on Name via mean
    out = (out.dropna(subset=["z"])
              .groupby("name", as_index=False)["z"].mean())
    meta = {"z_source": zkey}
    return out, "z", meta

def build_grid(df, xcol, ycol, nx, ny, margin):
    x = df[xcol].values; y = df[ycol].values
    xmin, xmax = x.min()-margin, x.max()+margin
    ymin, ymax = y.min()-margin, y.max()+margin
    gx = np.linspace(xmin, xmax, nx)
    gy = np.linspace(ymin, ymax, ny)
    GX, GY = np.meshgrid(gx, gy)
    return GX, GY, gx, gy, (xmin, xmax, ymin, ymax)

def interpolate(df, xcol, ycol, zcol, GX, GY, cfg):
    x = df[xcol].values; y = df[ycol].values; z = df[zcol].values
    method = cfg["interpolation"]["method"].lower()
    if method == "kriging":
        if len(df) < 5:
            fallback = cfg["interpolation"].get("griddata_fallback","linear")
            Z = griddata((x,y), z, (GX,GY), method=fallback)
            info = f"griddata fallback ({fallback}) due to sparse points"
            return Z, info
        kcfg = cfg["interpolation"]["kriging"]
        ok = OrdinaryKriging(
            x, y, z,
            variogram_model=kcfg.get("variogram_model","spherical"),
            nlags=kcfg.get("nlags",6),
            enable_plotting=kcfg.get("enable_plotting",False),
            coordinates_type="euclidean"
        )
        gx = np.unique(GX[0, :])
        gy = np.unique(GY[:, 0])
        Z, ss = ok.execute("grid", gx, gy)
        Z = np.asarray(Z)  # (ny, nx)
        info = "ordinary kriging"
        return Z, info
    else:
        Z = griddata((x,y), z, (GX,GY), method=method)
        return Z, f"griddata {method}"

def save_png(df, xcol, ycol, zcol, GX, GY, Z, cfg, out_png, title_suffix=""):
    # Single figure/axes; remove axes completely as requested
    fig, ax = plt.subplots(figsize=tuple(cfg["plot"]["figsize"]))
    ax.set_axis_off()

    levels = cfg["plot"]["levels"]
    zmin = float(np.nanmin(Z))
    zmax = float(np.nanmax(Z))

    def _levels_out_of_range(levels, zmin, zmax):
        return not (min(levels) < zmax and max(levels) > zmin)
    if _levels_out_of_range(levels, zmin, zmax):
        levels = np.linspace(zmin, zmax, cfg["plot"].get("auto_levels_n", 15))

    cmap = cfg["plot"].get("cmap", "gist_earth_r")

    cs = ax.contourf(GX, GY, Z, levels=levels, cmap=cmap)
    c = ax.contour(GX, GY, Z, colors="k", linewidths=0.4, levels=levels)
    if cfg["plot"].get("draw_contour_labels", True):
        ax.clabel(c, inline=True, fontsize=8, fmt="%.0f")

    # well markers
    sc = ax.scatter(
        df[xcol], df[ycol],
        s=26, marker="^",
        facecolors="white", edgecolors="k", linewidths=0.5
    )

    # labels (optional)
    if cfg["plot"].get("show_labels", True):
        name_col = cfg.get("csv_columns", {}).get("name", "name")
        dx = cfg["plot"].get("label_dx", 3)
        dy = cfg["plot"].get("label_dy", 3)
        for _, r in df.iterrows():
            ax.annotate(
                str(r[name_col]),
                (r[xcol], r[ycol]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=cfg["plot"].get("label_fontsize", 7),
                ha="left", va="bottom",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")]
            )

    ttl = cfg["plot"].get("title", "Pressure map")
    if title_suffix:
        ttl = f"{ttl} — {title_suffix}"
    ax.set_title(ttl, pad=10)
    cbar_label = cfg.get("z_label", zcol)
    fig.colorbar(cs, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_png, dpi=cfg["plot"].get("dpi",300))
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

def main():
    cfg_path = os.environ.get("CONFIG", "config/config_separated.yaml")
    cfg = load_cfg(cfg_path)

    # Read static & dynamic
    static_df, xcol, ycol, namecol, opt = read_static(cfg["input_static"], cfg)
    dynamic_df, zcol, meta = read_dynamic(cfg["input_dynamic"], cfg)

    # Merge on Name
    df = static_df.merge(dynamic_df, on="name", how="inner")

    # Build grid
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    margin = float(cfg.get("bounds_margin_units", 0.0))
    GX, GY, gx, gy, bounds = build_grid(df, xcol, ycol, nx, ny, margin)

    # Interpolate
    Z, info = interpolate(df, xcol, ycol, zcol, GX, GY, cfg)
    print(f"[INFO] Interpolation: {info}, points={len(df)}, grid=({ny},{nx}), z_source={meta['z_source']}")

    # Export
    outdir = pathlib.Path(cfg["output_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    stem = pathlib.Path(cfg["input_dynamic"]).stem
    if cfg["export"].get("png", True):
        png_path = outdir / f"{stem}_pressure_map.png"
        title_suffix = meta.get("z_source","").upper()
        save_png(df, xcol, ycol, zcol, GX, GY, Z, cfg, png_path, title_suffix=title_suffix)
        print(f"[OK] PNG saved: {png_path}")
    if cfg["export"].get("geotiff", True):
        tif_path = outdir / f"{stem}_pressure_map.tif"
        save_geotiff(Z, bounds, nx, ny, cfg["crs_epsg"], cfg["export"]["geotiff_nodata"], tif_path)
        print(f"[OK] GeoTIFF saved: {tif_path}")

if __name__ == "__main__":
    main()
