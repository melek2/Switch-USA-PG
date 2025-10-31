# %%
# DEPENDENCIES:
import os
from typing import Dict, List
from IPython.display import Markdown, display
import geopandas as gpd
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)  # Reset to default settings
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from typing import Dict, Optional
import scienceplots
import math

plt.style.use(['science','ieee'])
plt.rcParams['text.usetex'] = False

pd.options.display.float_format = '{:,.2f}'.format
import altair as alt
from pathlib import Path
import sys
sys.path.append('/Users/melek/Switch-USA-PG/MIP_results_comparison/notebooks/')

# %%
def load_csvs_as_vars(folder_path: str) -> None:
    """
    Reads all CSVs in `folder_path` and assigns each one
    to a global variable matching its filename (sans “.csv”).
    """
    for fname in os.listdir(folder_path):
        if not fname.endswith('.csv'):
            continue
        stem = os.path.splitext(fname)[0]
        path = os.path.join(folder_path, fname)
        globals()[stem] = pd.read_csv(path)


# %%
# KEY FOLDERS AND FILES
graph_tech_colors = '/Users/melek/Switch-USA-PG/switch/26-zone/in/2050/base_short/graph_tech_colors.csv'
conus_26_file = '/Users/melek/Switch-USA-PG/MIP_results_comparison/notebooks/conus_26z_latlon_simple.geojson'
# input_folder = "/Users/melek/Switch-USA-PG/switch/26-zone/in_test/2050/base_short"
input_folder = "/Users/melek/Switch-USA-PG/switch/26-zone/input_PM25/2050/base_short"
# output_folder = "/Users/melek/Switch-USA-PG/switch/26-zone/out/NoCluster"
output_folder = "/Users/melek/Switch-USA-PG/switch/26-zone/output_PM25/No_Policies"
# load inputs and outputs
load_csvs_as_vars(input_folder)
load_csvs_as_vars(output_folder)
# %% LOAD REGIONS DATA
load_areas = gpd.read_file(conus_26_file)
load_areas.rename(columns={'model_region':'load_zone'}, inplace=True)
load_areas['load_zone'].replace({'TRE_WEST':'TREW'}, inplace=True)
load_areas['centroid'] = load_areas['geometry'].apply(lambda x: x.centroid) #The centroids are later used to place pie charts

def translate_centroid(load_areas,load_zone, dx=0, dy=0):
    """
    Translate a centroid by dx and dy.
    """
    load_areas.loc[load_areas['load_zone']==
               load_zone,'centroid'] = gpd.points_from_xy(
                    x=load_areas.loc[load_areas['load_zone']==load_zone,'centroid'].x+dx,
                    y=load_areas.loc[load_areas['load_zone']==load_zone,'centroid'].y+dy
                                    )
    return load_areas

load_areas = translate_centroid(load_areas,'NWPP', dx=-3)
load_areas = translate_centroid(load_areas,'TREW', dy=-0.5)
load_areas = translate_centroid(load_areas,'MISW', dx=1)
load_areas = translate_centroid(load_areas,'SPPS', dx=0.5)
load_areas = translate_centroid(load_areas,'SPPN', dx=0.25)
load_areas = translate_centroid(load_areas,'MISE', dy=-0.25)
load_areas = translate_centroid(load_areas,'MISS', dy=-0.5)
load_areas = translate_centroid(load_areas,'NYCW', dy=-0.5,dx=0.5)

load_areas.set_index('load_zone', inplace=True, drop=False)
load_areas

# %% FUNCTION: DISPATCH ANNUAL SUMMARY 
def tech_generation_by_zone(dispatch_annual_summary):
    dispatch_annual_summary = dispatch_annual_summary[['gen_load_zone','gen_energy_source','period', 'Energy_GWh_typical_yr']]
    dispatch_annual_summary.rename(columns={'gen_load_zone' : 'load_zone'}, inplace=True)
    dispatch_annual_summary = dispatch_annual_summary.drop_duplicates().groupby(['load_zone', 'gen_energy_source', 'period'], sort=False, as_index=False).sum()

    total_by_zone= dispatch_annual_summary.pivot_table(index = 'load_zone', values = 'Energy_GWh_typical_yr', aggfunc=np.sum)
    total_by_zone.rename(columns={'Energy_GWh_typical_yr' : 'Total_by_zone'}, inplace=True)
    total_by_zone.reset_index(inplace=True)

    dispatch_annual_summary = pd.merge(left=dispatch_annual_summary, right=total_by_zone, on='load_zone')
    dispatch_annual_summary['Energy_relative'] = dispatch_annual_summary ['Energy_GWh_typical_yr']/dispatch_annual_summary['Total_by_zone']
    dispatch_annual_summary = dispatch_annual_summary.replace(np.nan, 0)
    dispatch_annual_summary = dispatch_annual_summary[dispatch_annual_summary.Total_by_zone != 0]

    return  dispatch_annual_summary
dispatch_annual_summary = tech_generation_by_zone(dispatch_zonal_annual_summary)

# %% TRANSMISSION DATA
def transmission_lines_df(BuildTx,transmission_lines, load_areas,interval_length_existing_cap=10,interval_length_new_cap=10):
    """
    Extracts transmission lines from the BuildTx DataFrame.
    """

    transmission_lines = pd.merge(
        BuildTx,
        transmission_lines,
        left_on='TRANS_BLD_YRS_1',
        right_on='TRANSMISSION_LINE',
        how='inner'
    )
    transmission_cap = transmission_lines[['TRANS_BLD_YRS_1','TRANS_BLD_YRS_2', 'trans_lz1', 'trans_lz2', 'existing_trans_cap', 'BuildTx']]
    transmission_cap = transmission_cap.rename(
        columns={'TRANS_BLD_YRS_2' : 'PERIOD'}
    )

    analysis_period = transmission_cap.PERIOD.unique()
    analysis_zones = set(list(transmission_cap.trans_lz1) + list(transmission_cap.trans_lz2))

    transmission_cap = transmission_cap.loc[((transmission_cap.trans_lz1.isin(analysis_zones)) | (transmission_cap.trans_lz2.isin(analysis_zones)))]
    transmission_cap = transmission_cap.loc[transmission_cap.PERIOD.isin(analysis_period)]

    transmission_cap['existing_trans_cap']= transmission_cap['existing_trans_cap']/1000 #Tranform capacity from MW to GW

    transmission_cap["BuildTx"].replace({'.': 0}, inplace=True)
    transmission_cap["BuildTx"] = transmission_cap["BuildTx"].astype(float)
    transmission_cap['BuildTx'] = transmission_cap['BuildTx']/1000 ## Tranform capacity from MW to GW
    
    # interval_length_existing_cap  = 10 # length of interval to plot for the existing lines
    # interval_length_new_cap = 10 # length of interval to plot for the new builtout
    transmission_cap['existing_lw'] = (transmission_cap['existing_trans_cap']/interval_length_existing_cap).apply(np.ceil)
    transmission_cap['build_lw'] = (transmission_cap['BuildTx']/interval_length_new_cap).apply(np.ceil)

    transmission_cap = pd.merge(left=transmission_cap, right=load_areas[['centroid']].reset_index().rename(columns={'load_zone' : 'trans_lz1'}), on='trans_lz1')
    transmission_cap.rename(columns={'centroid': 'coordinate_trans_lz1'}, inplace=True)
    transmission_cap = pd.merge(left=transmission_cap, right=load_areas[['centroid']].reset_index().rename(columns={'load_zone' : 'trans_lz2'}), on='trans_lz2')
    transmission_cap.rename(columns={'centroid': 'coordinate_trans_lz2'}, inplace=True)
    return transmission_cap
transmission_cap = transmission_lines_df(BuildTx=BuildTx,
                      transmission_lines=transmission_lines,
                      load_areas=load_areas)
#  %% LOAD GRAPH TECH COLORS
color_map = graph_tech_colors.set_index('gen_type')['color'].to_dict()


# %% Define the color map for each generation source
load_areas_color_map = {
    # West (WECC)
    'WECC_ID':   '#A6CEE3',  # light pacific blue
    'WEC_BANC':  '#FFE4B5',  # sandy desert beige
    'WEC_LADW':  '#E6E6FA',  # lavender sky
    'WECC_MT':   '#B5EAEA',  # mint glacier
    'WECC_CO':   '#87CEEB',  # pastel sky blue
    'WECC_AZ':   '#FFD8B1',  # peach canyon

    # Florida reliability council
    'FRCC':      '#B2DFDB',  # soft teal

    # New England / CT
    'NENG_CT':   '#D3D3D3',  # fog gray

    # Midwest (MIS)
    'MIS_IL':    '#C7EFCF',  # spring green
    'MIS_LMI':   '#CCFFCC',  # light lime
    'MIS_AMSO':  '#FFD1DC',  # blush pink
    'MIS_IA':    '#FFF5BA',  # pale yellow

    # New York zones
    'NY_Z_J':    '#F5B7B1',  # rose pastel
    'NY_Z_A':    '#FADBD8',  # soft coral

    # PJM
    'PJM_COMD':  '#E8DAEF',  # lilac haze
    'PJM_Dom':   '#D2B4DE',  # muted mauve
    'PJM_EMAC':  '#AFEEEE',  # light turquoise
    'PJM_AP':    '#FFE5B4',  # light apricot

    # SPP (plains)
    'SPP_N':     '#FFFACD',  # lemon chiffon
    'SPP_NEBR':  '#F5DEB3',  # wheat
    'SPP_SPS':   '#F0E68C',  # khaki

    # Southern
    'S_VACA':    '#F08080',  # light coral
    'S_C_KY':    '#FFA07A',  # salmon
    'S_SOU':     '#FFDAB9',  # peach puff

    # ERCOT
    'ERC_REST':  '#F5F5DC',  # beige
    'ERC_PHDL':  '#FFE4E1',  # misty rose
}
facecolors = load_areas['region'].map(load_areas_color_map)
# %%
def make_capacity_bins_and_handles(
    df,
    kind: str,
    lw_col: str,
    color: str,
    alpha: float,
    style: str,
    n_bins: int = 3
):
    """
    From df[kind] and df[lw_col], compute exactly n_bins capacity ranges
    and pick n_bins representative widths for the legend.
    """
    # only positive capacities
    mask = df[kind] > 0
    caps = df.loc[mask, kind].values
    widths = df.loc[mask, lw_col].values
    
    # sorted unique widths
    unique_lws = sorted(np.unique(widths))
    # pick n_bins widths evenly across unique_lws
    if len(unique_lws) >= n_bins:
        idx = np.linspace(0, len(unique_lws) - 1, n_bins, dtype=int)
        rep_lws = [unique_lws[i] for i in idx]
    else:
        # if fewer than n_bins widths, pad with the max width
        rep_lws = unique_lws + [unique_lws[-1]] * (n_bins - len(unique_lws))
    
    lo, hi = caps.min(), caps.max()
    edges = np.linspace(lo, hi, n_bins + 1)
    
    bin_labels = []
    handles = []
    for i, lw in enumerate(rep_lws):
        start, end = edges[i], edges[i + 1]
        label = f"{start:.0f}\u2013{end:.0f} GW"
        bin_labels.append(label)
        handles.append(
            Line2D([0], [0],
                   color=color,
                   lw=lw,
                   alpha=alpha,
                   linestyle=style)
        )
    
    return bin_labels, handles
# %%
# Create size bins and handles for the legend
def make_size_bins_and_handles(total_vals, max_radius, n_bins=3, marker_color='grey'):
    """
    Create n_bins size‐bins (in your data units) and legend handles
    with markers scaled to max_radius (in data units).

    Parameters
    ----------
    total_vals : array‐like
        All of your zone‐total values (e.g. dispatch_annual_summary['Total_by_zone']).
    max_radius : float
        The radius (in the same units your pie‐radius uses) that corresponds to the maximum total.
        In your code you do: radius = (val/max_tot)*2, so max_radius=2.
    n_bins : int
        How many circle‐sizes to show (3).
    marker_color : str
        The face‐color for each circle.

    Returns
    -------
    labels : list of str
    handles: list of Line2D
    """
    lo, hi = np.min(total_vals), np.max(total_vals)
    edges = np.linspace(lo, hi, n_bins + 1)

    # pick the midpoint of each bin for the "representative" total
    mids = [(edges[i] + edges[i+1]) / 2 for i in range(n_bins)]
    # convert those mids into radii (using your same pie‐scaling)
    radii = [(m / hi) * max_radius for m in mids]

    labels = [f"{edges[i]:.0f}\u2013{edges[i+1]:.0f} GWh"
              for i in range(n_bins)]
    handles = [
        Line2D([0], [0],
               marker='o',
               color='black',           # circle edge
               markerfacecolor=marker_color,
               markersize=r * 10,       # adjust 10→taste for legend size
               linestyle='') 
        for r in radii
    ]
    return labels, handles

# %%

def map_with_pies(
    load_areas: gpd.GeoDataFrame,
    dispatch_annual_summary: Optional[pd.DataFrame] = None,
    color_map: Optional[Dict[str, str]] = None,
    transmission_lines: Optional[pd.DataFrame] = None,
    dpi: int = 1000,
    include_legend: bool = True,
    include_region_labels: bool = False,
    include_region_colors: bool = True,
    load_areas_color_map: Optional[Dict[str, str]] = None,
    title: str = "Annual Generation by Load Zone and Source",
    save_path: Optional[str] = None,
) -> None:
    """
    Create a map with pie charts representing annual generation by source,
    overlaid on load‐zone polygons and (optional) transmission lines.

    Parameters
    ----------
    load_areas
        GeoDataFrame indexed by load_zone, with columns:
          - 'centroid' (Point)
          - optionally 'color' if include_region_colors is True
    dispatch_annual_summary
        DataFrame with columns ['load_zone','gen_energy_source',
        'Energy_GWh_typical_yr','Total_by_zone']
    color_map
        Mapping gen_energy_source → matplotlib color
    transmission_lines
        DataFrame with cols ['trans_lz1','trans_lz2','coordinate_trans_lz1',
        'coordinate_trans_lz2','existing_trans_cap','existing_lw',
        'BuildTx','build_lw']
    dpi
        Figure resolution
    save_path
        If given, save figure here
    """
    fig, ax = plt.subplots(dpi=dpi)
    xmin, ymin, xmax, ymax = load_areas.total_bounds

    # 1) water background
    ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                           facecolor='#ADD8E6', zorder=0))

    # 2) land fill
    if include_region_colors and load_areas_color_map:
        facecolors = (
            load_areas["region"]
            .map(load_areas_color_map)     # look up via region
            .fillna("white")               # anything missing -> white
        )
    else:
        facecolors = "white"
    load_areas.plot(ax=ax, facecolor=facecolors,
                    edgecolor='black', linewidth=0.1, zorder=1)

    # 3) transmission lines (behind pies)
    if transmission_lines is not None and not transmission_lines.empty:
        for kind, settings in [
            ('existing_trans_cap', ('red', 0.5, 'existing_lw', 2, '-')),
            ('BuildTx',           ('blue', 0.3, 'build_lw',    3,'--')),
        ]:
            color, alpha, lw_col, z, style = settings
            mask = transmission_lines[kind] > 0
            for _, row in transmission_lines.loc[mask].iterrows():
                x0, y0 = row['coordinate_trans_lz1'].x, row['coordinate_trans_lz1'].y
                x1, y1 = row['coordinate_trans_lz2'].x, row['coordinate_trans_lz2'].y
                ax.plot([x0, x1], [y0, y1],
                        lw=row[lw_col], linestyle=style, color=color,
                        alpha=alpha, zorder=z)
        # Create bins and handles for legends
        existing_bins, existing_handles = make_capacity_bins_and_handles(
            transmission_lines,
            kind='existing_trans_cap',
            lw_col='existing_lw',
            color='red',
            alpha=0.5,
            style='-'
        )

        new_bins, new_handles = make_capacity_bins_and_handles(
            transmission_lines,
            kind='BuildTx',
            lw_col='build_lw',
            color='blue',
            alpha=0.3,
            style='--'
        )


    # 4) pie charts
    if dispatch_annual_summary is not None and not dispatch_annual_summary.empty:
        radius_size_scale = 2
        max_tot = dispatch_annual_summary['Total_by_zone'].max()
        for zone, grp in dispatch_annual_summary.groupby('load_zone'):
            if zone not in load_areas.index:
                continue
            cent = load_areas.at[zone, 'centroid']
            df = grp[grp['Energy_GWh_typical_yr'] > 0]
            if df.empty:
                continue

            sizes = df['Energy_GWh_typical_yr']
            radius = (df['Total_by_zone'].max() / max_tot) * radius_size_scale
            ax.pie(
                sizes,
                colors=[color_map.get(src, 'grey') for src in df['gen_energy_source']],
                startangle=90,
                center=(cent.x, cent.y),
                radius=radius,
                wedgeprops={'edgecolor': 'black', 'linewidth': 0.05,'zorder': 5},
                # zorder=5,
            )

    ## 5) finalize axes
    ax.set_aspect('equal')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.linspace(xmin, xmax, 6))
    ax.set_yticks(np.linspace(ymin, ymax, 5))
    ax.set_xlabel("Longitude", fontsize=5)
    ax.set_ylabel("Latitude", fontsize=5)
    ax.tick_params(labelsize=5, length=1)
    ax.set_title(title, fontsize=7)

    # 6) legend
    
    if include_legend:
        if dispatch_annual_summary is not None and not dispatch_annual_summary.empty:
        # circle‐size legends
            totals = dispatch_annual_summary['Total_by_zone'].values
            circle_bins, circle_handles = make_size_bins_and_handles(
                total_vals=totals,
                max_radius=radius_size_scale,
                n_bins=3,
                marker_color='white' 
            )
            leg3 = ax.legend(
                circle_handles, circle_bins,
                title='Zone Total (GWh)',
                title_fontsize=4,
                fontsize=4,
                loc='upper center',
                bbox_to_anchor=(0.75, -0.2),
                labelspacing=3.5,
                handletextpad=4,
                ncol=1,
            )
            ax.add_artist(leg3)
        # transmission lines legends
        if transmission_lines is not None:
            # Existing‐Tx legend
            leg1 = ax.legend(
                existing_handles, existing_bins,
                title='Existing Tx (GW)',
                title_fontsize=4,
                fontsize=3.5,
                loc='upper center',
                bbox_to_anchor=(0.25, -0.25),
                ncol=3,
            )
            ax.add_artist(leg1)
            # New‐Tx legend
            leg2 = ax.legend(
                new_handles, new_bins,
                title='New Tx (GW)',
                title_fontsize=4,
                fontsize=3.5,
                loc='upper center',
                bbox_to_anchor=(0.25, -0.4),
                ncol=3,
            )
            ax.add_artist(leg2)
        if color_map:
        # Generation‐source legends
            patches = [
                    mpatches.Patch(
                        facecolor=color_map[k],
                        edgecolor='black',
                        linewidth=0.1, 
                        label=k
                    )
                    for k in color_map
            ]
            ax.legend(handles=patches,
                        loc='upper right',
                        bbox_to_anchor=(1.2, 1),
                        fontsize=4,
                        title='Generation Source',
                        title_fontsize=4)
            fig.subplots_adjust(right=1) 

    # 7) region labels
    if include_region_labels:
        for zone, row in load_areas.iterrows():
            pt = row['centroid']
            ax.text(pt.x, pt.y, zone,
                    ha='center', va='bottom', fontsize=3, zorder=6,fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()

map_with_pies(
    load_areas=load_areas,
    dispatch_annual_summary=dispatch_annual_summary,
    color_map=color_map,
    transmission_lines=transmission_cap,
    # save_path='Plots/basic_regions_map_No_Policy.png',
    title="EPA 26 Regions with Annual Generation by Source: No Policy",
    load_areas_color_map=load_areas_color_map,
    include_region_labels=True,
    include_legend=True,
)

# %%
