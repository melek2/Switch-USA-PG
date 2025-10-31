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
# from powergenome.financials import inflation_price_adjustment # does not work
CumulativeRateofInflation = 0.355 # from https://www.usinflationcalculator.com/ with 2011 (original year) to 2023 (base_financial_year)

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
table5_file      = Path('/Users/melek/Documents/Data/Emissions/NERC_Regions_Emission_Factors_ANL_2020_Table_5_clean.csv')               # From ANL-20/41 "TABLE 5 NERC regional generation-weighted average emission factors in g/kWh by fuel subtype and combustion technology"
mv_file          = Path('/Users/melek/Documents/Data/inmap/SRs/marginal_values_updated_110819/marginal_values_updated_110819.csv')
powerplant0_file = Path('/Users/melek/Documents/Data/DOE-OpenEnergy/power-plants0.csv')                                                 # From openenergyhub.ornl.gov "Power Plants - EIA"
IPM2NERC_file         = Path('/Users/melek/Documents/Data/ShapeFiles/IPM_to_NERC_mapping.csv')                                               # a csv file that maps each load zone to a NERC region. information from '/Users/melek/Desktop/Research/Capacity Expansion/Switch/Switch-USA-PG/MIP_results_comparison/case_settings/26-zone/settings-atb2023/model_definition.yml' line 82-119
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
EmissionFactorsNERC  = pd.read_csv(table5_file)
MV                   = pd.read_csv(mv_file)
IPM_to_NERC_mapping  = pd.read_csv(IPM2NERC_file)

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
# 5a) Map intensities to fuels.csv, archive, save
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
# 6) Emission Factors by generator
def build_EF_with_emissions(gen_info, IPM_to_NERC_mapping, table5):
    """
    Returns EF with columns:
      ['GENERATION_PROJECT','gen_tech','gen_energy_source','gen_load_zone','NERC_Region',
       'VOC_g_per_kWh','PM2.5_g_per_kWh','NOx_g_per_kWh']

    Rules (concise):
      - Non-emitting sources → 0 for all pollutants.
      - Petroleum Liquids with gen_energy_source == 'Distillate' → Oil/DFO.
      - Weighted averages by (region,fuel[,subfuel][,tech]) using Table 5 shares, with national fallbacks.
      - Tech inference from gen_tech: CC/GT/IC/ST.
    """
    # --- Validate ---
    gi = gen_info.copy()
    req_gi = {"GENERATION_PROJECT","gen_tech","gen_energy_source","gen_load_zone"}
    if not req_gi.issubset(gi.columns): raise ValueError("gen_info missing required columns.")

    if not {"gen_load_zone","NERC_Region"}.issubset(IPM_to_NERC_mapping.columns):
        raise ValueError("IPM_to_NERC_mapping must have ['gen_load_zone','NERC_Region']")

    pollutants = ["VOC_g_per_kWh","PM2.5_g_per_kWh","NOx_g_per_kWh"]
    req_t5 = {"NERC_region","Fuel_type","Fuel_subtype","Combustion_technology",
              "Fuel_type_share_pct","Fuel_subtype_share_pct","Combustion_share_pct", *pollutants}
    if not req_t5.issubset(table5.columns):
        missing = req_t5 - set(table5.columns)
        raise ValueError(f"table5 missing: {missing}")

    # --- Normalize Table 5 ---
    t5 = table5.copy()
    for c in ["NERC_region","Fuel_type","Fuel_subtype","Combustion_technology"]:
        t5[c] = t5[c].astype(str).str.strip().str.upper()
    for c in ["Fuel_type_share_pct","Fuel_subtype_share_pct","Combustion_share_pct", *pollutants]:
        t5[c] = pd.to_numeric(t5[c], errors="coerce")
    t5["__w"] = (
        t5["Fuel_type_share_pct"].fillna(100)
        * t5["Fuel_subtype_share_pct"].fillna(100)
        * t5["Combustion_share_pct"].fillna(100)
    )

    def wmeans(df):
        sw = np.nansum(df["__w"].to_numpy(dtype=float))
        out = {}
        for p in pollutants:
            v = df[p].to_numpy(dtype=float)
            if sw > 0 and np.isfinite(sw):
                out[p] = float(np.nansum(df["__w"].to_numpy()*v) / sw)
            else:
                v = v[~np.isnan(v)]
                out[p] = float(v.mean()) if v.size else np.nan
        return pd.Series(out)

    # Aggregates
    L1 = t5.groupby(["NERC_region","Fuel_type","Fuel_subtype","Combustion_technology"]).apply(wmeans).reset_index()
    L2 = t5.groupby(["NERC_region","Fuel_type","Fuel_subtype"]).apply(wmeans).reset_index()
    L3 = t5.groupby(["NERC_region","Fuel_type","Combustion_technology"]).apply(wmeans).reset_index()
    L4 = t5.groupby(["NERC_region","Fuel_type"]).apply(wmeans).reset_index()
    t5A = t5.assign(NERC_region="ALL")
    A1 = t5A.groupby(["NERC_region","Fuel_type","Fuel_subtype","Combustion_technology"]).apply(wmeans).reset_index()
    A2 = t5A.groupby(["NERC_region","Fuel_type","Fuel_subtype"]).apply(wmeans).reset_index()
    A3 = t5A.groupby(["NERC_region","Fuel_type","Combustion_technology"]).apply(wmeans).reset_index()
    A4 = t5A.groupby(["NERC_region","Fuel_type"]).apply(wmeans).reset_index()

    # Dict lookups
    def to_dict(df, cols):
        return {tuple(r[c] for c in cols): r[pollutants].to_dict() for _, r in df.iterrows()}
    D1, D2, D3, D4 = (to_dict(L1,["NERC_region","Fuel_type","Fuel_subtype","Combustion_technology"]),
                      to_dict(L2,["NERC_region","Fuel_type","Fuel_subtype"]),
                      to_dict(L3,["NERC_region","Fuel_type","Combustion_technology"]),
                      to_dict(L4,["NERC_region","Fuel_type"]))
    DA1, DA2, DA3, DA4 = (to_dict(A1,["NERC_region","Fuel_type","Fuel_subtype","Combustion_technology"]),
                          to_dict(A2,["NERC_region","Fuel_type","Fuel_subtype"]),
                          to_dict(A3,["NERC_region","Fuel_type","Combustion_technology"]),
                          to_dict(A4,["NERC_region","Fuel_type"]))

    # Fuel / subfuel / tech parsing
    def norm(s): return str(s).strip().lower() if pd.notna(s) else s
    fuel_map = {
        "naturalgas":"NG", "ng":"NG",
        "distillate":"OIL", "oil":"OIL",
        "coal":"COAL",
        "biomass":"BIOMASS", "waste_biomass":"BIOMASS",
        # non-emitters
        "electricity":None, "water":None, "wind":None, "solar":None,
        "uranium":None, "nuclear":None, "hydrogen":None, "heat":None
    }
    def subfuel_pref(gen_energy_source, _gen_tech):
        return "DFO" if norm(gen_energy_source) == "distillate" else None
    def tech_pref(gen_tech):
        s = norm(gen_tech or "")
        if "combined cycle" in s or " cc" in f" {s}" or "cc " in f"{s} ": return "CC"
        if "internal combustion" in s or "recip" in s or "engine" in s: return "IC"
        if "steam" in s or "conventional steam" in s: return "ST"
        if "combustion turbine" in s or " ct" in f" {s}" or "ct " in f"{s} " or ("gas turbine" in s and "steam" not in s): return "GT"
        return None

    gi = gi.merge(IPM_to_NERC_mapping[["gen_load_zone","NERC_Region"]], on="gen_load_zone", how="left")
    gi["__NER"]  = gi["NERC_Region"].astype(str).str.strip().str.upper()
    gi["__FUEL"] = gi["gen_energy_source"].map(lambda x: fuel_map.get(norm(x), None))
    gi["__SUB"]  = gi.apply(lambda r: subfuel_pref(r["gen_energy_source"], r["gen_tech"]), axis=1)
    gi["__TECH"] = gi["gen_tech"].map(tech_pref)

    # Resolver
    zero_vec = {p: 0.0 for p in pollutants}
    def resolve(n, f, sf, te):
        if f is None: return zero_vec
        if pd.notna(n):
            if sf and te and (n,f,sf,te) in D1:  return D1[(n,f,sf,te)]
            if sf and (n,f,sf) in D2:            return D2[(n,f,sf)]
            if te and (n,f,te) in D3:            return D3[(n,f,te)]
            if (n,f) in D4:                      return D4[(n,f)]
        if sf and te and ("ALL",f,sf,te) in DA1: return DA1[("ALL",f,sf,te)]
        if sf and ("ALL",f,sf) in DA2:           return DA2[("ALL",f,sf)]
        if te and ("ALL",f,te) in DA3:           return DA3[("ALL",f,te)]
        if ("ALL",f) in DA4:                     return DA4[("ALL",f)]
        return {p: np.nan for p in pollutants}

    resolved = [resolve(n,f,s,t) for n,f,s,t in zip(gi["__NER"],gi["__FUEL"],gi["__SUB"],gi["__TECH"])]
    for p in pollutants:
        gi[p] = [d[p] for d in resolved]

    EF = gi[[
        "GENERATION_PROJECT","gen_tech","gen_energy_source","gen_load_zone","NERC_Region",
        "VOC_g_per_kWh","PM2.5_g_per_kWh","NOx_g_per_kWh"
    ]].copy()

    # --- convert g/kWh → tonne/MMBtu for all pollutants ---
    # 1 g = 1e-6 tonne; 1 kWh = 0.003412 MMBtu
    conv_gpkwh_to_t_per_mmbtu = 1e-6 / 0.003412  # ≈ 0.00029308323563892143
    EF["VOC_t_per_MMBtu"]   = EF["VOC_g_per_kWh"]   * conv_gpkwh_to_t_per_mmbtu
    EF["PM2.5_t_per_MMBtu"] = EF["PM2.5_g_per_kWh"] * conv_gpkwh_to_t_per_mmbtu
    EF["NOx_t_per_MMBtu"]   = EF["NOx_g_per_kWh"]   * conv_gpkwh_to_t_per_mmbtu

    return EF
EF = build_EF_with_emissions(gen_info, IPM_to_NERC_mapping, EmissionFactorsNERC)
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
# 9a) Add MV average (PM2.5) and impute
target_mv_col = "MD_PM2.5_ground"

gen_all = add_mv_average_column(gen_all, MVdf, target_mv_col)
gen_all = impute_mv_by_zone_tech(gen_all, target_mv_col)

missing = gen_all[target_mv_col].isna().sum()
print(f"After imputation, missing {target_mv_col}: {missing}")
# Building gen_pm25_costs.csv, adjust MVs from 2011 dollars to to base_financial_year, save
gen_pm25_costs = pd.DataFrame({
    'GENERATION_PROJECT': gen_all['GENERATION_PROJECT'],
    'pm25_cost_dollar_per_ton': gen_all[target_mv_col],
    'gen_pm25_intensity_ton_per_MMBtu':EF['PM2.5_t_per_MMBtu']
})

# gen_pm25_costs['pm25_cost_dollar_per_ton'] = inflation_price_adjustment(
#     gen_pm25_costs['pm25_cost_dollar_per_ton'],
#     2011, base_financial_year
# )
gen_pm25_costs['pm25_cost_dollar_per_ton'] = gen_pm25_costs['pm25_cost_dollar_per_ton']*(1+CumulativeRateofInflation)

out_pm25 = input_path / "gen_pm25_costs.csv"
gen_pm25_costs.to_csv(out_pm25, index=False)
print(f"Saved {out_pm25} with {len(gen_pm25_costs):,} rows.")

# %% 
# 9b) Add MV average (NOx) and impute
target_mv_col = "MD_NOx_ground"

gen_all = add_mv_average_column(gen_all, MVdf, target_mv_col)
gen_all = impute_mv_by_zone_tech(gen_all, target_mv_col)

missing = gen_all[target_mv_col].isna().sum()
print(f"After imputation, missing {target_mv_col}: {missing}")
# %% 
# 9c) Add MV average (VOC) and impute
target_mv_col = "MD_VOC_ground"

gen_all = add_mv_average_column(gen_all, MVdf, target_mv_col)
gen_all = impute_mv_by_zone_tech(gen_all, target_mv_col)

missing = gen_all[target_mv_col].isna().sum()
print(f"After imputation, missing {target_mv_col}: {missing}")

# %% 
# 10a) Build gen_NOx_costs.csv, adjust MVs from 2011 dollars to to base_financial_year, save
gen_emission_costs = pd.DataFrame({
    'GENERATION_PROJECT': gen_all['GENERATION_PROJECT'],
    'pm25_cost_dollar_per_ton': gen_all['MD_PM2.5_ground'],
    'gen_pm25_intensity_ton_per_MMBtu':EF['PM2.5_t_per_MMBtu'],
    'NOx_cost_dollar_per_ton': gen_all['MD_NOx_ground'],
    'gen_NOx_intensity_ton_per_MMBtu':EF['NOx_t_per_MMBtu'],
    'VOC_cost_dollar_per_ton': gen_all['MD_VOC_ground'],
    'gen_VOC_intensity_ton_per_MMBtu':EF['VOC_t_per_MMBtu']
})

gen_emission_costs['pm25_cost_dollar_per_ton'] = gen_emission_costs['pm25_cost_dollar_per_ton']*(1+CumulativeRateofInflation)
gen_emission_costs['NOx_cost_dollar_per_ton'] = gen_emission_costs['NOx_cost_dollar_per_ton']*(1+CumulativeRateofInflation)
gen_emission_costs['VOC_cost_dollar_per_ton'] = gen_emission_costs['VOC_cost_dollar_per_ton']*(1+CumulativeRateofInflation)


out_emissions_file = input_path / "gen_emission_costs.csv"
gen_emission_costs.to_csv(out_emissions_file, index=False)
print(f"Saved {out_emissions_file} with {len(gen_emission_costs):,} rows.")
# %%
