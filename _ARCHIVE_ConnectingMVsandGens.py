import pandas as pd
import os
import ast
import geopandas as gpd
from shapely.geometry import Point
from Functions.gen_info_manipulation import get_plants_coordinates
from Functions.LoadMVs import load_MVs_as_gdf, load_MVs_as_df


MV_file = '/Users/melek/Documents/Data/inmap/SRs/marginal_values_updated_110819/marginal_values_updated_110819.csv'
gen_info_file = '/Users/melek/Switch-USA-PG/switch/26-zone/in_test/2050/base_short/gen_info2.csv'
powerplant0_file = '/Users/melek/Documents/Data/DOE-OpenEnergy/power-plants0.csv'

gen_info = get_plants_coordinates(gen_info_file, powerplant0_file)
MVgdf = load_MVs_as_gdf(MV_file)
MVdf = load_MVs_as_df(MV_file)





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

# Apply row‐wise and make a new column "cell_IDs"
gen_info["cell_IDs"] = gen_info.apply(
    lambda r: find_cell_ids_for_row(r, MVgdf),
    axis=1
)

import pandas as pd

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

gen_info = add_mv_average_column(gen_info, MVdf, "MD_PM2.5_ground")


def impute_mv_by_zone_tech(gen_info: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    For any NaN in gen_info[column], replace it with the average value from the “closest” rows:
      1. First look for rows with the same gen_load_zone AND same gen_tech (non‐NaN in `column`).
      2. If none exist, look for rows with the same gen_load_zone (any gen_tech).
      3. If still none, look for rows with the same gen_tech (any gen_load_zone).
      4. If still none, use the global average over gen_info[column].

    This mutates (or fills) gen_info[column] in place and returns the DataFrame.
    """
    # 1) Mask of rows where column is not NaN
    valid_mask = gen_info[column].notna()
    valid = gen_info.loc[valid_mask, :]
    if valid.empty:
        # No valid values at all → nothing to impute
        return gen_info

    # Precompute the global mean of the column
    global_mean = valid[column].mean()

    # 2) Iterate only over the rows where column is NaN
    na_idx = gen_info.index[gen_info[column].isna()]

    for i in na_idx:
        zone = gen_info.at[i, "gen_load_zone"]
        tech = gen_info.at[i, "gen_tech"]

        # (a) same zone & same tech
        subset = valid.loc[
            (valid["gen_load_zone"] == zone) & (valid["gen_tech"] == tech),
            column
        ]
        if not subset.empty:
            fill_val = subset.mean()
            rule = "same zone & same tech"

        else:
            # (b) same zone (any tech)
            subset = valid.loc[valid["gen_load_zone"] == zone, column]
            if not subset.empty:
                fill_val = subset.mean()
                rule = "same zone (any tech)"
            else:
                # (c) same tech (any zone)
                subset = valid.loc[valid["gen_tech"] == tech, column]
                if not subset.empty:
                    fill_val = subset.mean()
                    rule = "same zone (any tech)"
                else:
                    # (d) fallback to global mean
                    fill_val = global_mean
                    rule = "global mean fallback"
        gen_info.at[i, column] = fill_val
        print(
                    f"Plant {i} (zone={zone}, tech={tech}) did not have a {column} and was replaced by average from {rule}  "
                    # f" with value {fill_val:.6f}"
                )
    return gen_info

gen_info = impute_mv_by_zone_tech(gen_info, "MD_PM2.5_ground")


gen_pm25_costs = pd.DataFrame({
    'GENERATION_PROJECT': gen_info['GENERATION_PROJECT'],
    'pm25_cost_dollar_per_ton': gen_info['MD_PM2.5_ground']
})

# Get the directory of the gen_info_file
gen_info_dir = os.path.dirname(gen_info_file)
# Build the output file path
output_csv = os.path.join(gen_info_dir, "gen_pm25_costs.csv")
# Save the DataFrame
gen_pm25_costs.to_csv(output_csv, index=False)
fuels_file = '/Users/melek/Switch-USA-PG/switch/26-zone/in_test/2050/base_short/fuels.csv'
fuels = pd.read_csv(fuels_file)
