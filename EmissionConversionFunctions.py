"""
Functions to add data to Switch inputs.
"""
# coppied from /Users/melek/SRMs/Functions/gen_info_manipulation.py
import os
import pandas as pd
import ast
import os
import geopandas as gpd
from shapely.geometry import Point

def extract_plant_ids(x):
    """
    - If x is a Python list (e.g. ['59817_BATT1', '60654_GEN02', ...]), 
      convert each element to str, split on '_', take the left part, 
      then cast to float→int.
    - If x is a string that literal‐evaluates to a list, do the same.
    - Otherwise (float, int, <NA>), return x unchanged.
    """
    # Case A: x is already a list
    if isinstance(x, list):
        out = []
        for item in x:
            first_chunk = str(item).split('_', 1)[0]
            try:
                out.append(int(float(first_chunk)))
            except ValueError:
                continue
        return out

    # Case B: x is a string that looks like a Python list
    if isinstance(x, str):
        stripped = x.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    out = []
                    for item in parsed:
                        first_chunk = str(item).split('_', 1)[0]
                        try:
                            out.append(int(float(first_chunk)))
                        except ValueError:
                            continue
                    return out
            except (ValueError, SyntaxError):
                return x

    # Case C: anything else (float, int, <NA>, etc.), leave unchanged
    return x


def map_ids_to_coords(x, code_to_lon, code_to_lat):
    """
    Given x = gen_info["plant_id"], return a dict with lists of matching longitudes
    and latitudes from the provided lookup dicts.
    """
    # If x is a list, collect lon/lat for each element that exists in the dict
    if isinstance(x, list):
        lons = [code_to_lon[pid] for pid in x if pid in code_to_lon]
        lats = [code_to_lat[pid] for pid in x if pid in code_to_lat]
        return {"Longitude": lons, "Latitude": lats}

    # If x is a single integer or Int64, attempt a lookup
    if pd.notna(x):
        lon = code_to_lon.get(x)
        lat = code_to_lat.get(x)
        return {
            "Longitude": [lon] if lon is not None else [],
            "Latitude":  [lat] if lat is not None else []
        }

    # If x is NA or something else, return empty lists
    return {"Longitude": [], "Latitude": []}


def get_plants_coordinates(gen_info_path, power_plants_path, save_as_csv=False):
    """
    Read gen_info CSV and power_plants CSV, extract plant_id(s) from 'unit_id_pg',
    map them to coordinates, append 'plant_id', 'Longitude', 'Latitude' columns,
    save the augmented DataFrame as 'gen_info2.csv' in the same directory as gen_info_path,
    and return the resulting DataFrame.
    
    Parameters:
    -----------
    gen_info_path : str
        Full path to the gen_info CSV file.
    power_plants_path : str
        Full path to the power_plants CSV file.

    Returns:
    --------
    pd.DataFrame
        The augmented gen_info DataFrame with new columns: 'plant_id', 'Longitude', 'Latitude'.
    """
    # 1) Read input files
    gen_info = pd.read_csv(gen_info_path)
    powerplants = pd.read_csv(power_plants_path)

    # 2) Extract plant_id from 'unit_id_pg' where it's a list or list‐string
    is_list_or_list_str = gen_info['unit_id_pg'].apply(
        lambda x: isinstance(x, list)
                  or (isinstance(x, str)
                      and x.strip().startswith('[')
                      and x.strip().endswith(']'))
    )
    gen_info.loc[is_list_or_list_str, 'plant_id'] = (
        gen_info.loc[is_list_or_list_str, 'unit_id_pg']
                .apply(extract_plant_ids)
    )

    # 3) Build lookup dicts for Longitude and Latitude by Plant Code
    code_to_lon = powerplants.set_index("Plant Code")["Longitude"].to_dict()
    code_to_lat = powerplants.set_index("Plant Code")["Latitude"].to_dict()

    # 4) Map plant_id(s) to coordinates
    coords_series = gen_info["plant_id"].apply(
        lambda x: map_ids_to_coords(x, code_to_lon, code_to_lat)
    )
    coords_df = pd.DataFrame(coords_series.tolist())

    # 5) Concatenate the new coordinate columns onto gen_info
    gen_info = pd.concat([gen_info, coords_df], axis=1)

    # 6) Save to gen_info2.csv in same directory as gen_info_path
    out_dir = os.path.dirname(os.path.abspath(gen_info_path))
    out_path = os.path.join(out_dir, "gen_info2.csv")
    if save_as_csv is True:
        gen_info.to_csv(out_path, index=False)
    gen_info = gen_info.loc[:, ~gen_info.columns.duplicated(keep='first')]
    return gen_info

# Copied from /Users/melek/SRMs/Functions/LoadMVs.py
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

def load_MVs_as_gdf(csv_path):
    """
    Reads a marginal values CSV file and returns a GeoDataFrame with columns:
    - cell_ID: grid cell identifier
    - geometry: polygon of the cell in WGS84 coordinates (EPSG:4326)
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Create geometry in original projection (Lambert Conformal Conic 2SP)
    df['geometry'] = df.apply(
        lambda row: box(row['Location_W'], row['Location_S'], row['Location_E'], row['Location_N']),
        axis=1
    )
    
    # Define source CRS
    lcc_crs = (
        "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 "
        "+x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs"
    )
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df[['cell_ID', 'geometry']],
        crs=lcc_crs
    )
    
    # Reproject to WGS84
    gdf = gdf.to_crs(epsg=4326)
    return gdf

def load_MVs_as_df(csv_path):
    """
    Reads a marginal values CSV file and returns a DataFrame with all columns except 'geometry' collumns:
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    df.drop(columns=['Location_W', 'Location_S', 'Location_E', 'Location_N'], inplace=True)
    
    return df

# from /Users/melek/Documents/GitHub/Switch-USA-PG/ConnectingMVsandGens.py

def find_cell_ids_for_row(row, cells):
    # 1) Grab lon/lat; if they’re strings, parse into actual Python lists
    lons = row["Longitude"]
    lats = row["Latitude"]

    if isinstance(lons, str):
        try:
            lons = ast.literal_eval(lons)
        except (ValueError, SyntaxError):
            return []   # can’t parse → skip
    if isinstance(lats, str):
        try:
            lats = ast.literal_eval(lats)
        except (ValueError, SyntaxError):
            return []

    # 2) Now check that both are lists of the same length
    if not isinstance(lons, (list, tuple)) or not isinstance(lats, (list, tuple)):
        return []
    if len(lons) != len(lats) or len(lons) == 0:
        return []

    # 3) Build a tiny GeoDataFrame of all points in this row
    pts = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(lons, lats)],
        crs="EPSG:4326"
    )

    # 4) Spatial‐join: each point picks up any polygon’s cell_ID it falls in
    joined = gpd.sjoin(
        pts,
        cells[["cell_ID", "geometry"]],
        how="left",
        predicate="within"
    )

    # 5) Return unique, non‐null cell_IDs
    return joined["cell_ID"].dropna().unique().tolist()

def add_mv_average_column(gen_info: pd.DataFrame, MVdf: pd.DataFrame, mv_column: str) -> pd.DataFrame:
    """
    For each row in `gen_info`, uses its `cell_IDs` list to look up `mv_column` in MVdf
    and computes the average over all matching cells. Creates (or overwrites) a column
    in gen_info named exactly `mv_column` containing those averages.

    Parameters
    ----------
    gen_info : pd.DataFrame
        Must have a column "cell_IDs" where each entry is a Python list of integers.
    MVdf : pd.DataFrame
        Must have columns "cell_ID" and `mv_column`. 
    mv_column : str
        Name of the column in MVdf to average (e.g. "MD_PM2.5_ground").

    Returns
    -------
    pd.DataFrame
        The same gen_info DataFrame, with a new column `mv_column` containing the
        row‐wise mean of MVdf[mv_column] over all IDs in gen_info["cell_IDs"].
    """
    # Create a Series indexed by cell_ID for fast lookup
    mv_series = MVdf.set_index("cell_ID")[mv_column]

    def _row_avg(cell_list):
        # If not a list or empty, return NaN
        if not isinstance(cell_list, (list, tuple)) or len(cell_list) == 0:
            return float("nan")
        # Filter out any IDs not present in mv_series
        valid_ids = [cid for cid in cell_list if cid in mv_series.index]
        if not valid_ids:
            return float("nan")
        return mv_series.loc[valid_ids].mean()

    gen_info[mv_column] = gen_info["cell_IDs"].apply(_row_avg)
    return gen_info
def impute_mv_by_zone_tech(gen_info: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    For any NaN in gen_info[column], replace it with the average value from the “closest” rows:
      1. same zone & same tech
      2. same zone & same gen_energy_source
      3. same zone (any tech)
      4. same tech (any zone)
      5. global mean fallback

    Prints each imputation and, at the end, how many times each rule fired.
    """
    
    valid = gen_info.loc[gen_info[column].notna(), :]
    if valid.empty:
        return gen_info

    # Precompute means for fast lookups
    global_mean = valid[column].mean()
    mean_zone_tech = valid.groupby(["gen_load_zone", "gen_tech"], dropna=False)[column].mean()
    # Only build the zone+energy_source map if the column exists
    has_energy_source = "gen_energy_source" in gen_info.columns
    if has_energy_source:
        mean_zone_fuel = valid.groupby(["gen_load_zone", "gen_energy_source"], dropna=False)[column].mean()
    mean_zone = valid.groupby(["gen_load_zone"], dropna=False)[column].mean()
    mean_tech = valid.groupby(["gen_tech"], dropna=False)[column].mean()

    na_idx = gen_info.index[gen_info[column].isna()]

    rule_counts = {
        "same zone & same tech":            0,
        "same zone & same energy source":   0,
        "same zone (any tech)":             0,
        "same tech (any zone)":             0,
        "global mean fallback":             0,
    }

    for i in na_idx:
        zone = gen_info.at[i, "gen_load_zone"]
        tech = gen_info.at[i, "gen_tech"]
        fuel = gen_info.at[i, "gen_energy_source"] if has_energy_source else None

        # 1) same zone & same tech
        fill_val = mean_zone_tech.get((zone, tech), None)
        rule = None

        # 2) same zone & same gen_energy_source
        if fill_val is None and has_energy_source and pd.notna(fuel):
            fill_val = mean_zone_fuel.get((zone, fuel), None)
            if fill_val is not None:
                rule = "same zone & same energy source"

        # 3) same zone (any tech)
        if fill_val is None:
            fill_val = mean_zone.get(zone, None)
            if fill_val is not None:
                rule = "same zone (any tech)"

        # 4) same tech (any zone)
        if fill_val is None:
            fill_val = mean_tech.get(tech, None)
            if fill_val is not None:
                rule = "same tech (any zone)"

        # 5) global mean fallback
        if fill_val is None or pd.isna(fill_val):
            fill_val = global_mean
            rule = "global mean fallback"

        # If rule still None, it means the first rule hit (zone+tech)
        if rule is None:
            rule = "same zone & same tech"

        gen_info.at[i, column] = fill_val
        rule_counts[rule] += 1
        print(f"Plant {i} (zone={zone}, tech={tech}"
              + (f", fuel={fuel}" if has_energy_source else "")
              + f") imputed via {rule}")

    # summary
    print("\nImputation summary:")
    for rule, cnt in rule_counts.items():
        print(f"  {rule:30s}: {cnt}")

    return gen_info
