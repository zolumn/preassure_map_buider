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

def read_static(path, sep=",", x="x", y="y", name="well_name", idcol="well_id"):
    df = pd.read_csv(path, delimiter=sep)
    df.columns = df.columns.str.lower()
    need = [idcol, x, y]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Static file missing columns: {missing}")
    return df[[idcol, x, y] + ([name] if name in df.columns else [])].dropna()


def read_csv(csv_path, cols, sep):
    df = pd.read_csv(csv_path, delimiter=sep)
    df = df.rename(columns=str.lower)
  
    # маппинг имен
    xcol, ycol, zcol = [cols[k].lower() for k in ("x","y","z")]
    name_col = cols.get("name")
    if name_col:
        name_col = name_col.lower()
    else:
        # если явно не задано — берем самый первый столбец как имя
        name_col = df.columns[0]

    need = [xcol, ycol, zcol, name_col]
    print(f"[DEBUG] Need columns: {need}")
    print(f"[DEBUG] DF columns: {df.columns}")

    for c in [xcol, ycol, zcol]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in CSV")

    # если колонки имени нет — создадим безопасные метки
    if name_col not in df.columns:
        print(f"[WARN] Name column '{name_col}' not found; will autogenerate labels")
        name_col = "_well_name_autogen"
        df[name_col] = [f"P{i+1}" for i in range(len(df))]

    df = df.dropna(subset=[xcol, ycol, zcol])

    # усредняем дубликаты координат; имя берём первое (или уникальные имена склеиваем)
    agg = {
        zcol: "mean",
        name_col: (lambda s: s.iloc[0] if s.notna().any() else "")
        # если хотите все имена в одной точке склеивать:
        # name_col: (lambda s: ", ".join(sorted(map(str, pd.unique(s)))))
    }
    df = (df.groupby([xcol, ycol], as_index=False)
            .agg(agg)
            .sort_values([xcol, ycol]))
    return df, xcol, ycol, zcol, name_col

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
    fig, ax = plt.subplots(figsize=tuple(cfg["plot"]["figsize"]))

    # Убираем оси с координатами и цифры
    ax.set_axis_off()
    
    levels = cfg["plot"]["levels"]
    zmin = float(np.nanmin(Z))
    zmax = float(np.nanmax(Z))
    print(f"[DEBUG] Z range: [{zmin:.3f}, {zmax:.3f}]")
    
    def _levels_out_of_range(levels, zmin, zmax):
        return not (min(levels) < zmax and max(levels) > zmin)
    
    if _levels_out_of_range(levels, zmin, zmax):
        print("[WARN] Provided levels are outside Z range; auto-adjusting.")
        levels = np.linspace(zmin, zmax, 15)
        
    # 1) геологическая палитра по умолчанию или берем из конфига
    cmap = cfg["plot"].get("cmap", "gist_earth_r")

    cs = plt.contourf(GX, GY, Z, levels=levels, cmap=cmap)
    c = plt.contour(GX, GY, Z, colors="k", linewidths=0.4, levels=levels)
    plt.clabel(c, inline=True, fontsize=8, fmt="%.0f")

    # 2) треугольники на скважинах
    # sc = plt.scatter(df[xcol], df[ycol], c=df[zcol], s=22, marker="^",
    #                  edgecolors="k", linewidths=0.4)
    sc = plt.scatter(
        df[xcol], df[ycol],
        s=26, marker="^",
        facecolors="white", edgecolors="k", linewidths=0.5
    )

    # 3) подписи имен скважин
    # имя колонки можно задать в конфиге csv_columns.name.
    # если нет — берём самый первый столбец CSV.    
    name_col = cfg.get("csv_columns", {}).get("name")
    if name_col is None:
        name_col = df.columns[0]

    # небольшой сдвиг подписи, чтобы не попадать ровно на маркер
    dx = cfg["plot"].get("label_dx", 3)   # в тех же единицах, что и координаты
    dy = cfg["plot"].get("label_dy", 3)

    for _, r in df.iterrows():
        plt.annotate(
            str(r[name_col]),
            (r[xcol], r[ycol]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=cfg["plot"].get("label_fontsize", 7),
            ha="left", va="bottom",
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")]
        )

    plt.title("Pressure map (interpolated)")
    plt.colorbar(cs, label=zcol)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
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
    cfg_path = os.environ.get("CONFIG", "config/config.yaml")
    cfg = load_cfg(cfg_path)
    csv = cfg["input_csv"]
    outdir = pathlib.Path(cfg["output_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    df, xcol, ycol, zcol, name_col = read_csv(csv, cfg["csv_columns"], cfg["csv_sep"])

    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    margin = float(cfg["bounds_margin_units"])
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
