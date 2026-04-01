"""
Full WB2 ERA5 NetCDF download: time-varying variables use monthly shards, then merge
to a single YYYY.nc (matches process_one_step_data.py expectations).

Flow per (variable, year):
  1. If {year}.nc exists -> skip.
  2. Else download missing months to {year}_{MM}.nc (skip non-empty existing shards).
  3. When all 12 shards exist, merge -> {year}.nc via atomic replace, then remove shards.

CLI:
  python download_full_data.py              # download + merge as needed
  python download_full_data.py --merge-only # only merge dirs with 12 monthly shards & no YYYY.nc
"""
import argparse
import calendar
import os

import xarray as xr
from tqdm import tqdm

save_dir = "/home/zhangbo/stormer_data/wb2_nc"
file = "1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"
years = [2018, 2019, 2020]


def _month_time_slice(year: int, month: int):
    last = calendar.monthrange(year, month)[1]
    return slice(f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last:02d}")


def _monthly_shard_path(save_dir_var: str, year: int, month: int) -> str:
    return os.path.join(save_dir_var, f"{year}_{month:02d}.nc")


def _shard_looks_ok(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _merge_monthly_shards(save_dir_var: str, year: int, out_path: str) -> None:
    month_paths = [_monthly_shard_path(save_dir_var, year, m) for m in range(1, 13)]
    missing = [p for p in month_paths if not _shard_looks_ok(p)]
    if missing:
        raise RuntimeError(f"incomplete monthly shards: {missing}")

    tmp_path = out_path + ".merging.tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    combined = xr.open_mfdataset(
        month_paths,
        combine="by_coords",
        parallel=False,
        coords="minimal",
        data_vars="minimal",
    )
    try:
        combined = combined.sortby("time")
        try:
            combined.to_netcdf(tmp_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
    finally:
        combined.close()

    os.replace(tmp_path, out_path)

    for p in month_paths:
        os.remove(p)


def merge_pending_only() -> None:
    """Merge any variable/year that has 12 non-empty monthly shards but no {year}.nc yet."""
    merged = 0
    for name in sorted(os.listdir(save_dir)):
        save_dir_var = os.path.join(save_dir, name)
        if not os.path.isdir(save_dir_var):
            continue
        for year in years:
            out_path = os.path.join(save_dir_var, f"{year}.nc")
            if os.path.exists(out_path):
                continue
            month_paths = [_monthly_shard_path(save_dir_var, year, m) for m in range(1, 13)]
            if not all(_shard_looks_ok(p) for p in month_paths):
                continue
            print(f"Merging {name}/{year} ...", flush=True)
            try:
                _merge_monthly_shards(save_dir_var, year, out_path)
                print(f"  OK -> {out_path}", flush=True)
                merged += 1
            except Exception as e:
                print(f"  FAIL {name}/{year}: {e}", flush=True)
    print(f"Merge-only done. Merged {merged} year file(s).", flush=True)


def main_download() -> None:
    os.makedirs(save_dir, exist_ok=True)
    print("Connecting to GCS...")
    ds = xr.open_zarr("gs://weatherbench2/datasets/era5/" + file)
    variables = list(ds.keys())
    print(f"Found {len(variables)} variables, downloading years: {years}")

    for var in tqdm(variables, desc="variables"):
        ds_var = ds[[var]]
        if len(ds_var.dims) < 3:
            out_path = os.path.join(save_dir, f"{var}.nc")
            if os.path.exists(out_path):
                tqdm.write(f"  Skip {var} (exists)")
                continue
            tqdm.write(f"  Saving {var}...")
            try:
                ds_var.to_netcdf(out_path)
            except Exception as e:
                tqdm.write(f"  [ERROR] {var}: {e}")
        else:
            save_dir_var = os.path.join(save_dir, var)
            os.makedirs(save_dir_var, exist_ok=True)
            for year in years:
                out_path = os.path.join(save_dir_var, f"{year}.nc")
                if os.path.exists(out_path):
                    tqdm.write(f"  Skip {var}/{year} (exists)")
                    continue

                for month in range(1, 13):
                    shard = _monthly_shard_path(save_dir_var, year, month)
                    if _shard_looks_ok(shard):
                        continue
                    tqdm.write(f"  {var}/{year} month {month:02d}...")
                    try:
                        ds_var.sel(time=_month_time_slice(year, month)).to_netcdf(shard)
                    except Exception as e:
                        tqdm.write(f"  [ERROR] {var}/{year}-{month:02d}: {e}")
                        if os.path.exists(shard):
                            os.remove(shard)

                try:
                    _merge_monthly_shards(save_dir_var, year, out_path)
                    tqdm.write(f"  Done {var}/{year} (merged from monthly shards)")
                except Exception as e:
                    tqdm.write(f"  [WARN] {var}/{year} merge deferred: {e}")

    print("All done!")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge completed monthly shards into YYYY.nc (no GCS download)",
    )
    args = parser.parse_args()
    if args.merge_only:
        merge_pending_only()
    else:
        main_download()


if __name__ == "__main__":
    main()
