# %% 
# 0) Imports (deduped) + small helpers
import os
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd

from shapely.geometry import box
from EmissionConversionFunctions import (
    get_plants_coordinates, load_MVs_as_gdf, load_MVs_as_df,
    find_cell_ids_for_row, add_mv_average_column, impute_mv_by_zone_tech
)
# from powergenome.financials import inflation_price_adjustment

def to_fraction_from_percent(s: pd.Series) -> pd.Series:
    """'87.5%' or 87.5 -> 0.875; blanks -> NaN."""
    return (
        s.astype(str).str.strip().str.rstrip('%').replace({'': np.nan})
        .astype(float) / 100.0
    )
# %% 
# 1) Paths (alternatives are commented out to flip quickly)
# input_path = Path('/Users/melek/Desktop/Research/Capacity Expansion/Air quality/Inputs/switch_inputs_foresight/base_20_week_2050/2050/base_20_week')
input_path = Path('/Users/melek/Desktop/Research/Air quality/Inputs/switch_inputs_foresight/base_20_week_2045/2045/base_20_week_test')

fuels_file       = input_path / 'fuels.csv'
gen_info_file    = input_path / 'gen_info2.csv'
financials_file  = input_path / 'financials.csv'

cerf_siting_file = Path('/Users/melek/Desktop/Research/Capacity Expansion/Siting/conus26/cerf_inputs/outputs/cerf_candidate_sites.csv') # From [cerf26.ipnyb](/Users/melek/Desktop/Research/Capacity Expansion/Siting/conus26/cerf26.ipynb)
table2_file      = Path('/Users/melek/Documents/Data/Emissions/Energy_Conversion_Efficiecies_ANL_2020_Table2.csv')                      # From ANL-20/41 "TABLE 2 National and regional energy conversion efficiencies by fuel subtype and combustion technology."
table4_file      = Path('/Users/melek/Documents/Data/Emissions/Emission_Factors_ANL_2020_Table_4.csv')                                  # From ANL-20/41 "TABLE 4 National generation-weighted average emission factors in g/kWh by fuel subtype and combustion technology"
mv_file          = Path('/Users/melek/Documents/Data/inmap/SRs/marginal_values_updated_110819/marginal_values_updated_110819.csv')
powerplant0_file = Path('/Users/melek/Documents/Data/DOE-OpenEnergy/power-plants0.csv')                                                 # From openenergyhub.ornl.gov "Power Plants - EIA"
# %% 
# 2) Load core inputs
# switch files
fuels       = pd.read_csv(fuels_file)
gen_info    = pd.read_csv(gen_info_file)
financials  = pd.read_csv(financials_file)

# other files
cerf_candidate_sites = pd.read_csv(cerf_siting_file) if cerf_siting_file.exists() else None
EmissionFactors      = pd.read_csv(table4_file)
HeatRates            = pd.read_csv(table2_file)
MV                   = pd.read_csv(mv_file)

# scalar base year
base_financial_year = int(pd.Series(financials['base_financial_year']).iloc[0])
print(f"Loaded. base_financial_year={base_financial_year}, gen_info={len(gen_info):,} rows, fuels={len(fuels):,} rows.")
# %% 
# 3) Tidy ANL Table 4 shares to fractions
for col in ['Fuel_type_share', 'Fuel_subtype_share', 'Combustion_technology_share']:
    if col in EmissionFactors.columns:
        EmissionFactors[col] = to_fraction_from_percent(EmissionFactors[col])
EmissionFactors['weight'] = (
    EmissionFactors.get('Fuel_subtype_share', 1.0) *
    EmissionFactors.get('Combustion_technology_share', 1.0)
)

# pollutants to compute
pollutants = ['NOx','SOx','PM2.5','PM10','VOC','CO','CH4','N2O']

# weighted-average EF (g/kWh) by Fuel_type
ef_per_fuel = (
    EmissionFactors
    .groupby('Fuel_type', as_index=False)
    .apply(lambda df: (df[pollutants].mul(df['weight'], axis=0).sum() / df['weight'].sum()))
    .reset_index(drop=True)
)
ef_per_fuel.head()
# %% 
# 4) Tidy ANL Table 2 efficiencies
# Expect columns: 'Fuel type', 'Fuel subtype share (%)', 'Combustion tech share (%)', 'National'
if {'Fuel subtype share (%)','Combustion tech share (%)'}.issubset(HeatRates.columns):
    HeatRates = HeatRates.copy()
    HeatRates['weight'] = HeatRates['Fuel subtype share (%)'] * HeatRates['Combustion tech share (%)']
else:
    # fallback if already fraction columns exist without (%)
    HeatRates['weight'] = HeatRates.get('Fuel subtype share', 1.0) * HeatRates.get('Combustion tech share', 1.0)

efficiency_by_fuel = (
    HeatRates
    .groupby('Fuel type', as_index=False)
    .apply(lambda df: (df[['National']].mul(df['weight'], axis=0).sum() / df['weight'].sum()))
    .reset_index(drop=True)
    .rename(columns={'National': 'Efficiency'})
)

# Efficiency given in percent; convert to fraction
efficiency_by_fuel['Efficiency'] = efficiency_by_fuel['Efficiency'] / 100.0
efficiency_by_fuel.head()
# %% 
# 5) Join EF + Efficiency and compute intensities (t/MMBTU)
BTU_PER_KWH = 3412.0

ef_hr = ef_per_fuel.merge(
    efficiency_by_fuel[['Fuel type','Efficiency']],
    left_on='Fuel_type', right_on='Fuel type', how='left'
)

for p in pollutants:
    col = f"f_{p.lower().replace('.','')}_intensity"
    # g/kWh * (kWh_e/kWh_th) / 3412 Btu/kWh = g/Btu = t/MMBTU numerically
    ef_hr[col] = ef_hr[p] * ef_hr['Efficiency'] / BTU_PER_KWH

intensity_cols = [c for c in ef_hr.columns if c.endswith('_intensity')]
lookup = ef_hr.set_index('Fuel_type')[intensity_cols]
lookup.head()
# %% 
# 6) Map intensities to fuels.csv, archive, save
fuel_to_ef = {
    'Coal':          'Coal',
    'Naturalgas':    'NG',
    'Distillate':    'Oil',
    'Waste_biomass': 'Biomass',
    'Fuel':          'NG',      # generic Fuel -> NG
    'Uranium':       None,      # zero
    'Hydrogen':      None,      # zero
}

fuels_out = fuels.copy()
for col in intensity_cols:
    mapping = {
        fuel: (lookup.at[key, col] if (key is not None and key in lookup.index) else 0.0)
        for fuel, key in fuel_to_ef.items()
    }
    fuels_out[col] = fuels_out['fuel'].map(mapping)

# archive then write
fuels_archive = input_path / 'fuels_archive.csv'
# fuels.to_csv(fuels_archive, index=False)
# fuels_out.to_csv(fuels_file, index=False)
print(f"fuels.csv updated with {len(intensity_cols)} intensity columns. Archive -> {fuels_archive}")
# %%
# 7) Exclude CERF candidate projects from gen_info (if present)
if cerf_candidate_sites is not None:
    before = len(gen_info)
    gen_info = gen_info[~gen_info['GENERATION_PROJECT'].isin(cerf_candidate_sites['GENERATION_PROJECT'])]
    print(f"Filtered gen_info: {before} -> {len(gen_info)} (removed CERF candidates)")
else:
    print("No CERF candidate sites file found; keeping gen_info as-is.")
# %% 
# 8) Attach coordinates and MV grid
# Note: get_plants_coordinates reads from file path, not df
gen_info = get_plants_coordinates(str(gen_info_file), str(powerplant0_file))

MVgdf = load_MVs_as_gdf(str(mv_file))
MVdf  = load_MVs_as_df(str(mv_file))

# cell IDs for each set
gen_info['cell_IDs'] = gen_info.apply(lambda r: find_cell_ids_for_row(r, MVgdf), axis=1)

if cerf_candidate_sites is not None:
    cerf_candidate_sites['cell_IDs'] = cerf_candidate_sites.apply(lambda r: find_cell_ids_for_row(r, MVgdf), axis=1)
    gen_all = pd.concat([gen_info, cerf_candidate_sites], ignore_index=True)
else:
    gen_all = gen_info.copy()

print(f"Combined generators count: {len(gen_all):,}")
# %% 
# 9) Add MV average (PM2.5) and impute
target_mv_col = "MD_PM2.5_ground"

gen_all = add_mv_average_column(gen_all, MVdf, target_mv_col)
gen_all = impute_mv_by_zone_tech(gen_all, target_mv_col)

missing = gen_all[target_mv_col].isna().sum()
print(f"After imputation, missing {target_mv_col}: {missing}")
# %% 
# 10) Build gen_pm25_costs.csv, adjust MVs from 2011 dollars to to base_financial_year, save
gen_pm25_costs = pd.DataFrame({
    'GENERATION_PROJECT': gen_all['GENERATION_PROJECT'],
    'pm25_cost_dollar_per_ton': gen_all[target_mv_col]
})

gen_pm25_costs['pm25_cost_dollar_per_ton'] = inflation_price_adjustment(
    gen_pm25_costs['pm25_cost_dollar_per_ton'],
    2011, base_financial_year
)

out_pm25 = input_path / "gen_pm25_costs.csv"
# gen_pm25_costs.to_csv(out_pm25, index=False)
print(f"Saved {out_pm25} with {len(gen_pm25_costs):,} rows.")
