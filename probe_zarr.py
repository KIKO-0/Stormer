import xarray as xr

def check_zarr(file_name):
    print(f"Connecting to WeatherBench2 GCS: {file_name}...")
    try:
        ds = xr.open_zarr('gs://weatherbench2/datasets/era5/' + file_name)
        vars = list(ds.keys())
        print(f"✅ Successfully opened {file_name}.")
        print(f"Total variables: {len(vars)}")
        
        target_var = 'mean_surface_latent_heat_flux'
        if target_var in vars:
            print(f"✅ Found missing variable: {target_var}")
        else:
            print(f"❌ Variable {target_var} is STILL MISSING in this dataset.")
            
        return vars
    except Exception as e:
        print(f"Failed to open {file_name}: {e}")
        return []

# Check the exact file mentioned in README
ds1_vars = check_zarr('1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr')

# Let's also check another common ERA5 dataset on WB2 without the date suffix
ds2_vars = check_zarr('1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr')
