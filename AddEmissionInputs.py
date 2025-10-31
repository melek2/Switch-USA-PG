# %%
# Load dependencies
import pandas as pd
import os,sys
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from EmissionConversionFunctions import get_plants_coordinates, load_MVs_as_gdf, load_MVs_as_df,find_cell_ids_for_row, add_mv_average_column,impute_mv_by_zone_tech  
from powergenome.financials import inflation_price_adjustment
# %%
# Set the target year for USD conversion
# switch input Files
input_path = '/Users/melek/Desktop/Research/Capacity Expansion/Air quality/Inputs/switch_inputs_foresight/base_20_week_2050/2050/base_20_week'
input_path = '/Users/melek/Desktop/Research/Capacity Expansion/Air quality/Inputs/switch_inputs_foresight/base_10_week_2050/2050/base_short'
fuels_file = os.path.join(input_path, 'fuels.csv')
fuels = pd.read_csv(fuels_file)
gen_info_file = os.path.join(input_path, 'gen_info2.csv')
gen_info = pd.read_csv(gen_info_file)
financials_file = os.path.join(input_path, 'financials.csv')
financials = pd.read_csv(financials_file)
# NOTE THAT THIS CANDIDATES_SITES.CSV IS DIFFERENT THAN THE ONE GENERATED IN PG_TO_SWITCH.PY, THIS
# THIS FILE IS GENERATED FROM [cerf26.ipnyb](/Users/melek/Desktop/Research/Capacity Expansion/Siting/conus26/cerf26.ipynb)
cerf_siting_file = '/Users/melek/Desktop/Research/Capacity Expansion/Siting/conus26/cerf_inputs/outputs/cerf_candidate_sites.csv'
cerf_candidate_sites = pd.read_csv(cerf_siting_file)

# ANL-20/41 "TABLE 4 National generation-weighted average emission factors in g/kWh by fuel subtype and combustion technology"
Table_4 = '/Users/melek/Documents/Data/Emissions/Emission_Factors_ANL_2020_Table_4.csv'
EmissionFactors = pd.read_csv(Table_4)
# ANL-20/41 "TABLE 2 National and regional energy conversion efficiencies by fuel subtype and combustion technology."
Table_2 = '/Users/melek/Documents/Data/Emissions/Energy_Conversion_Efficiecies_ANL_2020_Table2.csv'
HeatRates = pd.read_csv(Table_2)
# Converting Share columns to float
for col in ['Fuel_type_share','Fuel_subtype_share','Combustion_technology_share']:
    EmissionFactors[col] = (
        EmissionFactors[col]
        .str.rstrip('%')     # remove trailing “%”
        .astype(float)       # to float like 87.50
        .div(100)            # to 0.8750
    )
# Marginal Values
MV_file = '/Users/melek/Documents/Data/inmap/SRs/marginal_values_updated_110819/marginal_values_updated_110819.csv'
MV = pd.read_csv(MV_file)
# From openenergyhub.ornl.gov "Power Plants - EIA"
powerplant0_file = '/Users/melek/Documents/Data/DOE-OpenEnergy/power-plants0.csv'


# %% Financial information
base_financial_year = int(financials['base_financial_year'])

# %%
if cerf_candidate_sites is not None:
        # if candidate sites file exists, filter gen_info to exlude those projects
    gen_info = gen_info[~gen_info['GENERATION_PROJECT'].isin(cerf_candidate_sites['GENERATION_PROJECT'])]
# %%
# Add latitude and longitude coordinates to each generator using EIA power plant data
gen_info = get_plants_coordinates(gen_info_file, powerplant0_file)

# Load marginal values as a GeoDF for spatial operations
MVgdf = load_MVs_as_gdf(MV_file)

# Load marginal values as a regular DF
MVdf = load_MVs_as_df(MV_file)
# %% 
cerf_candidate_sites
cerf_candidate_sites["cell_IDs"] = cerf_candidate_sites.apply(
    lambda r: find_cell_ids_for_row(r, MVgdf),
    axis=1
)
# %%
# Assign cell IDs to each generator based on spatial location and marginal value grid
gen_info["cell_IDs"] = gen_info.apply(
    lambda r: find_cell_ids_for_row(r, MVgdf),
    axis=1
)
# %%
gen_info = pd.concat([gen_info, cerf_candidate_sites], ignore_index=True)

# %%
# creating gen_pm25_costs.csv
# Add average marginal PM2.5 cost for each generator based on its cell ID
gen_info = add_mv_average_column(gen_info, MVdf, "MD_PM2.5_ground")

# Impute missing marginal values by using the average for each zone and technology
gen_info = impute_mv_by_zone_tech(gen_info, "MD_PM2.5_ground")
# %%
# DataFrame with generation project names and their associated PM2.5 MD costs
gen_pm25_costs = pd.DataFrame({
    'GENERATION_PROJECT': gen_info['GENERATION_PROJECT'],
    'pm25_cost_dollar_per_ton': gen_info['MD_PM2.5_ground']
})

# Convert PM2.5 costs to USD2020 using inflation adjustment
gen_pm25_costs['pm25_cost_dollar_per_ton'] = inflation_price_adjustment(gen_pm25_costs['pm25_cost_dollar_per_ton'],2011,2020)

# output CSV file path
gen_pm25_costs_csv = os.path.join(input_path, "gen_pm25_costs.csv")

# Save as CSV file
gen_pm25_costs.to_csv(gen_pm25_costs_csv, index=False)

# %%
# Archive of original fuels.csv before modifications
fuels_file_archive = os.path.join(input_path, 'fuels_archive.csv')
fuels.to_csv(fuels_file_archive, index=False)

# Update fuels.csv with emission factors
fuel_to_ef = {
    'Coal':          'Coal',
    'Naturalgas':    'NG',
    'Distillate':    'Oil',
    'Waste_biomass': 'Biomass',
    'Fuel':          'NG',      # assume generic “Fuel” is NG
    'Uranium':       None,      # no EF will get 0
    'Hydrogen':      None,      # no EF will get 0
}
fuels = pd.read_csv(fuels_file)
# --- 1) Compute efficiency fraction from Table 2 (HeatRates) ---
HeatRates['weight'] = (
    HeatRates['Fuel subtype share (%)']
  * HeatRates['Combustion tech share (%)']
)
hr_per_fuel = (
    HeatRates
      .groupby('Fuel type')
      .apply(
         lambda df: (
             df[['National']]
               .multiply(df['weight'], axis=0)
               .sum()
           / df['weight'].sum()
         )
      )
      .reset_index()
      .rename(columns={'National': 'Efficiency'})
)
# convert percent to fraction
hr_per_fuel['Efficiency'] /= 100


# --- 2) Compute weighted-average emission factors (g/kWh) from Table 4 ---
EmissionFactors['weight'] = (
    EmissionFactors['Fuel_subtype_share']
  * EmissionFactors['Combustion_technology_share']
)
pollutants = ['NOx','SOx','PM2.5','PM10','VOC','CO','CH4','N2O']
ef_per_fuel = (
    EmissionFactors
      .groupby('Fuel_type')
      .apply(
         lambda df: (
             df[pollutants]
               .multiply(df['weight'], axis=0)
               .sum()
           / df['weight'].sum()
         )
      )
      .reset_index()
)


# --- 3) Merge the two and compute t pollutant /MMBTU ---
# 3412 Btu per kWh
Btu_per_kWh = 3412.  

# join by fuel key
ef_hr = ef_per_fuel.merge(
    hr_per_fuel[['Fuel type','Efficiency']],
    left_on = 'Fuel_type',
    right_on= 'Fuel type',
    how='left'
)

# for each pollutant, compute:
#   f_p (g/kWh) * Efficiency (kWh_e/kWh_th) / 3412 (Btu_th/kWh_th)
#   → gives g pollutant per Btu_th, which = t pollutant per MMBTU_th
for p in pollutants:
    out_col = f"f_{p.lower().replace('.','')}_intensity"
    ef_hr[out_col] = ef_hr[p] * ef_hr['Efficiency'] / Btu_per_kWh


# --- 4) Map those back onto fuels DF ---
# make a lookup table
lookup = ef_hr.set_index('Fuel_type')[
    [c for c in ef_hr.columns if c.endswith('_intensity')]
]

# assume fuels['fuel'] matches the names in lookup.index
intensity_cols = [c for c in ef_hr.columns if c.endswith('_intensity')]
lookup = ef_hr.set_index('Fuel_type')[intensity_cols]

# now map each fuels.fuel → lookup key → the intensities (or 0)
for col in intensity_cols:
    mapping = {
        fuel: (lookup.at[key, col] if (key is not None and key in lookup.index) else 0.0)
        for fuel, key in fuel_to_ef.items()
    }
    fuels[col] = fuels['fuel'].map(mapping)

# dropping pollutants that are not needed for now
# fuels=fuels[['fuel','co2_intensity','upstream_co2_intensity','f_pm25_intensity']]

# --- 5) Save the updated fuels.csv ---
fuels.to_csv(fuels_file, index=False)

# %%
