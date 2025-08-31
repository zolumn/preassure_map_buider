import os, sys, pathlib, warnings, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pykrige.ok import OrdinaryKriging
from rasterio.transform import from_origin
import rasterio

warnings.filterwarnings("ignore", category=UserWarning)

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_csv(csv_path, cols):
    print(f"{csv_path}")
    df = pd.read_csv(csv_path)
    df = df.rename(columns=str.lower)
  
    # маппинг имён
    xcol, ycol, zcol = [cols[k].lower() for k in ("Bt_Coord_X","Bt_Coord_Y","Pres(2434)")]
    need = [xcol, ycol, zcol]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in CSV")
    df = df.dropna(subset=[xcol,ycol,zcol])
    # усредняем дубликаты координат
    df = (df.groupby([xcol,ycol], as_index=False)[zcol].mean()
            .sort_values([xcol,ycol]))
    return df, xcol, ycol, zcol

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
            # слишком мало точек — fallback
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
        # kriging "grid" принимает 1D массивы gx, gy
        gx = np.unique(GX[0, :])
        gy = np.unique(GY[:, 0])
        Z, ss = ok.execute("grid", gx, gy)
        Z = np.asarray(Z)  # (ny, nx)
        info = "ordinary kriging"
        return Z, info
    else:
        Z = griddata((x,y), z, (GX,GY), method=method)
        return Z, f"griddata {method}"

def save_png(df, xcol, ycol, zcol, GX, GY, Z, cfg, out_png):
    plt.figure(figsize=tuple(cfg["plot"]["figsize"]))
    levels = cfg["plot"]["levels"]
    cmap = cfg["plot"].get("cmap","viridis")
    cs = plt.contourf(GX, GY, Z, levels=levels, cmap=cmap)
    c = plt.contour(GX, GY, Z, colors="k", linewidths=0.4, levels=levels)
    plt.clabel(c, inline=True, fontsize=8, fmt="%.0f")
    sc = plt.scatter(df[xcol], df[ycol], c=df[zcol], s=18, edgecolors="k", linewidths=0.3)
    plt.title("Pressure map (interpolated)")
    plt.colorbar(cs, label=zcol)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

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
    cfg_path = os.environ.get("CONFIG", "config.yaml")
    cfg = load_cfg(cfg_path)
    csv = cfg["input_csv"]
    outdir = pathlib.Path(cfg["output_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    df, xcol, ycol, zcol = read_csv(csv, cfg["csv_columns"])
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    margin = float(cfg["grid"]["bounds_margin_units"])
    GX, GY, gx, gy, bounds = build_grid(df, xcol, ycol, nx, ny, margin)
    Z, info = interpolate(df, xcol, ycol, zcol, GX, GY, cfg)
    print(f"[INFO] Interpolation: {info}, points={len(df)}, grid=({ny},{nx})")

    # Экспорт
    stem = pathlib.Path(csv).stem
    if cfg["export"].get("png", True):
        png_path = outdir / f"{stem}_pressure_map.png"
        save_png(df, xcol, ycol, zcol, GX, GY, Z, cfg, png_path)
        print(f"[OK] PNG saved: {png_path}")
    if cfg["export"].get("geotiff", True):
        tif_path = outdir / f"{stem}_pressure_map.tif"
        save_geotiff(Z, bounds, nx, ny, cfg["crs_epsg"], cfg["export"]["geotiff_nodata"], tif_path)
        print(f"[OK] GeoTIFF saved: {tif_path}")

if __name__ == "__main__":
    main()
