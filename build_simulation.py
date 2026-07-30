#!/usr/bin/env python3
"""
Human Survivability Risk — Simulation Engine
Extends the XLSX model (2025–2200) to 2000–2500 with non-linear feedback cascades.
Generates GeoJSON timeseries for the dashboard.
"""

import json, math, sys
import pandas as pd
import numpy as np

XLSX_PATH = "../.reasonix/attachments/clipboard-20260730-155658.255727-000034.xlsx"
OUT_JSON = "data/survivability_timeseries.json"
OUT_SUMMARY = "data/continental_summary.json"
OUT_META = "data/region_meta.json"

# ── Load XLSX ──────────────────────────────────────────────
df = pd.read_excel(XLSX_PATH, sheet_name="Global Master Data")
cont_df = pd.read_excel(XLSX_PATH, sheet_name="Continental Summary")

# ── Extract region metadata ────────────────────────────────
regions = df[['Country', 'Region / Zone', 'Continent', 'Lat', 'Lon']].drop_duplicates()
region_meta = {}
for _, r in regions.iterrows():
    key = r['Region / Zone']
    region_meta[key] = {
        'country': r['Country'],
        'continent': r['Continent'],
        'lat': float(r['Lat']),
        'lon': float(r['Lon'])
    }

# ── Build per-region timeseries dict (2025–2200) ──────────
cols = ['MAT (°C)', 'Max WBT (°C)', 'Days >= 110°F', 'PM2.5 (µg/m³)',
        'Water Stress Index', 'Mod. Poverty (%)', 'Ext. Poverty (%)',
        'Adaptive Cap (%)', 'Survivability Index (SI)', 'Population (M)',
        'Cumul. Excess Mortality (M)', 'Habitability Status']

ts = {}  # ts[region][year] = { ... }
for _, row in df.iterrows():
    r = row['Region / Zone']
    y = int(row['Year'])
    if r not in ts:
        ts[r] = {}
    ts[r][y] = {c: row[c] for c in cols}
    ts[r][y]['Year'] = y

# ── Backfill 2000–2020 ─────────────────────────────────────
def backfill_2000_2020(ts):
    """Linearly extrapolate backwards from 2025 values using
    historical warming rate ~0.18°C/decade (IPCC AR6 observed)."""
    for region, data in ts.items():
        v2025 = data[2025]
        for y in range(2000, 2025, 5):
            frac = (2025 - y) / 25.0  # 0 at 2025, 1 at 2000
            entry = {}
            entry['Year'] = y
            entry['MAT (°C)'] = round(v2025['MAT (°C)'] - 0.45 * frac, 1)
            entry['Max WBT (°C)'] = round(v2025['Max WBT (°C)'] - 0.35 * frac, 1)
            entry['Days >= 110°F'] = round(v2025['Days >= 110°F'] * (1 - 0.25 * frac))
            entry['PM2.5 (µg/m³)'] = round(v2025['PM2.5 (µg/m³)'] * (1 - 0.15 * frac), 1)
            entry['Water Stress Index'] = round(v2025['Water Stress Index'] * (1 - 0.10 * frac), 1)
            # Poverty higher in past, adaptive capacity lower
            entry['Mod. Poverty (%)'] = round(min(95, v2025['Mod. Poverty (%)'] * (1 + 0.3 * frac)), 1)
            entry['Ext. Poverty (%)'] = round(min(70, v2025['Ext. Poverty (%)'] * (1 + 0.4 * frac)), 1)
            entry['Adaptive Cap (%)'] = round(max(5, v2025['Adaptive Cap (%)'] * (1 - 0.3 * frac)), 1)
            # Better SI in past (cooler)
            entry['Survivability Index (SI)'] = round(min(1.0, v2025['Survivability Index (SI)'] + 0.08 * frac), 3)
            entry['Population (M)'] = round(v2025['Population (M)'] * (1 - 0.15 * frac), 1)
            entry['Cumul. Excess Mortality (M)'] = 0.0
            # Determine status
            si = entry['Survivability Index (SI)']
            if si >= 0.75: entry['Habitability Status'] = 'Safe / Low Risk'
            elif si >= 0.50: entry['Habitability Status'] = 'Moderate Stress'
            elif si >= 0.25: entry['Habitability Status'] = 'High / Critical Risk'
            else: entry['Habitability Status'] = 'Uninhabitable Zone'
            data[y] = entry

# ── Extend 2200–2500 with non-linear feedback ──────────────
def extend_2200_2500(ts):
    """Extend the model using four compounding feedback loops from the PDF:
    1. Aerosol Termination Shock (+0.6°C jump by 2220)
    2. Permafrost Methane Release (accelerates post-2150, adds ~0.05°C/decade)
    3. Amazon Dieback (~90 Gt CO2 pulse around 2150)
    4. Albedo Loss (Arctic ice gone, accelerating warming)
    
    By 2500: equatorial/tropical zones are unsurvivable.
    Only boreal refugia (Canada, Siberia) remain habitable.
    """
    for region, data in ts.items():
        v2200 = data[2200]
        meta = region_meta[region]
        lat = meta['lat']
        
        # Boreal refugia (>50°N) stay habitable but degrade slowly
        is_boreal = lat > 50
        # Mid-latitude (30-50°N/S) moderate degradation
        is_mid = 30 <= abs(lat) <= 50
        # Tropical/subtropical (<30°) severe degradation
        is_tropical = abs(lat) < 30
        
        for y in range(2205, 2505, 5):
            steps = (y - 2200) / 5
            entry = {}
            entry['Year'] = y
            
            # Non-linear acceleration factor
            # After 2250, feedback loops compound aggressively
            if y <= 2250:
                accel = 1.0 + 0.02 * steps  # mild acceleration
            elif y <= 2350:
                accel = 1.0 + 0.02 * 10 + 0.05 * (steps - 10)  # permafrost methane kicks in
            else:
                accel = 1.0 + 0.02 * 10 + 0.05 * 20 + 0.10 * (steps - 30)  # full cascade
            
            # Aerosol termination shock: rapid jump around 2210-2230
            if 2210 <= y <= 2230:
                shock = (y - 2210) / 20.0 * 0.8
            elif y > 2230:
                shock = 0.8
            else:
                shock = 0.0
            
            # Boreal: slow degradation
            if is_boreal:
                mat_rate = 0.06 * accel
                wbt_rate = 0.04 * accel
                si_decay = 0.006 * accel
                pop_decay = 0.98  # population stable or grows (refugia migration)
            elif is_mid:
                mat_rate = 0.12 * accel
                wbt_rate = 0.10 * accel
                si_decay = 0.015 * accel
                pop_decay = 0.92
            else:  # tropical
                mat_rate = 0.18 * accel
                wbt_rate = 0.16 * accel
                si_decay = 0.025 * accel
                pop_decay = 0.82
            
            entry['MAT (°C)'] = round(v2200['MAT (°C)'] + mat_rate * steps + shock * (0.5 if is_tropical else 0.2), 1)
            entry['Max WBT (°C)'] = round(v2200['Max WBT (°C)'] + wbt_rate * steps + shock * (0.6 if is_tropical else 0.25), 1)
            
            # Days >= 110°F saturates at 250-300 for hottest regions
            base_days = v2200['Days >= 110°F']
            max_days = 280 if is_tropical else (180 if is_mid else 40)
            entry['Days >= 110°F'] = round(min(max_days, base_days + 3.5 * steps * accel))
            
            # PM2.5: declines after 2200 as population collapses, but aerosol termination shock may spike
            if y <= 2250:
                entry['PM2.5 (µg/m³)'] = round(v2200['PM2.5 (µg/m³)'] * (1 + 0.01 * steps), 1)
            else:
                entry['PM2.5 (µg/m³)'] = round(v2200['PM2.5 (µg/m³)'] * max(0.5, 1 - 0.015 * (steps - 10)), 1)
            
            # Water stress saturates
            entry['Water Stress Index'] = round(min(100, v2200['Water Stress Index'] + 0.3 * steps * accel), 1)
            
            # Poverty: worsens then irrelevant (population collapse)
            entry['Mod. Poverty (%)'] = round(min(98, v2200['Mod. Poverty (%)'] + 0.4 * steps * accel), 1)
            entry['Ext. Poverty (%)'] = round(min(75, v2200['Ext. Poverty (%)'] + 0.3 * steps * accel), 1)
            
            # Adaptive capacity collapses in hardest-hit regions
            entry['Adaptive Cap (%)'] = round(max(2, v2200['Adaptive Cap (%)'] - 0.8 * steps * accel), 1)
            
            # Survivability Index
            raw_si = v2200['Survivability Index (SI)'] - si_decay * steps
            entry['Survivability Index (SI)'] = round(max(0.01, min(1.0, raw_si)), 3)
            
            # Population: collapses in tropics, stable/grows in boreal refugia
            entry['Population (M)'] = round(v2200['Population (M)'] * (pop_decay ** steps), 1)
            
            # Cumulative excess mortality: accelerates
            base_mort = v2200['Cumul. Excess Mortality (M)']
            mort_per_step = entry['Population (M)'] * 0.03 * accel
            entry['Cumul. Excess Mortality (M)'] = round(base_mort + mort_per_step * steps, 1)
            
            # Habitability status
            si = entry['Survivability Index (SI)']
            if si >= 0.75: entry['Habitability Status'] = 'Safe / Low Risk'
            elif si >= 0.50: entry['Habitability Status'] = 'Moderate Stress'
            elif si >= 0.25: entry['Habitability Status'] = 'High / Critical Risk'
            else: entry['Habitability Status'] = 'Uninhabitable Zone'
            
            data[y] = entry

# ── Run extensions ─────────────────────────────────────────
backfill_2000_2020(ts)
extend_2200_2500(ts)

# ── Generate GeoJSON FeatureCollection per year ────────────
# Map each region to its country for the GeoJSON properties
# We'll use country-level granularity for the map
country_region_map = {}
for r, meta in region_meta.items():
    country_region_map[meta['country']] = r

# Build per-year GeoJSON feature properties
all_years = set()
for data in ts.values():
    for y in data:
        all_years.add(y)
all_years = sorted(all_years)

# Per-country-per-year data for the dashboard
country_year_data = {}
for region, data in ts.items():
    country = region_meta[region]['country']
    if country not in country_year_data:
        country_year_data[country] = {}
    for y, vals in data.items():
        if y not in country_year_data[country]:
            country_year_data[country][y] = vals
        else:
            # If multiple regions per country (e.g. USA), take the worst SI
            existing = country_year_data[country][y]
            if vals['Survivability Index (SI)'] < existing['Survivability Index (SI)']:
                country_year_data[country][y] = vals

# Also handle multi-region countries by creating region entries
# USA: Texas Coastal Plain + Sonoran Desert/Southwest
# Australia: Southeast Basin + Central Outback
# Brazil: Amazon Basin Fringe + Northeast Sertão

# Build the full timeseries output
output = {
    'years': all_years,
    'regions': {},
    'countries': country_year_data
}

for region, data in ts.items():
    output['regions'][region] = {
        'meta': region_meta[region],
        'timeseries': {str(y): vals for y, vals in sorted(data.items())}
    }

# ── Continental summary ────────────────────────────────────
continental_summary = {}
for _, row in cont_df.iterrows():
    y = int(row['Year'])
    c = row['Continent']
    if y not in continental_summary:
        continental_summary[y] = {}
    continental_summary[y][c] = {
        'avgMAT': float(row['Avg MAT (°C)']),
        'avgWBT': float(row['Avg Max WBT (°C)']),
        'avgDays110': float(row['Avg Days >= 110°F']),
        'avgPM25': float(row['Avg PM2.5']),
        'avgSI': float(row['Avg SI']),
        'totalPop': float(row['Total Pop (M)']),
        'cumulMortality': float(row['Cumul. Excess Mortality (M)']),
        'pctUninhabitable': float(row['% Uninhabitable Zones'])
    }

# ── SI Tier definitions ────────────────────────────────────
tier_defs = [
    {'tier': 'Safe / Low Risk', 'min': 0.75, 'max': 1.0, 'color': '#3aa89e', 'label': 'Safe'},
    {'tier': 'Moderate Stress', 'min': 0.50, 'max': 0.75, 'color': '#d4a84b', 'label': 'Moderate'},
    {'tier': 'High / Critical Risk', 'min': 0.25, 'max': 0.50, 'color': '#e07040', 'label': 'High Risk'},
    {'tier': 'Uninhabitable Zone', 'min': 0.0, 'max': 0.25, 'color': '#c04060', 'label': 'Critical'},
    {'tier': 'Extinction Zone', 'min': -1, 'max': 0.05, 'color': '#6b3fa0', 'label': 'Extinction'}
]

def si_to_tier_color(si):
    if si >= 0.75: return '#3aa89e'
    elif si >= 0.50: return '#d4a84b'
    elif si >= 0.25: return '#e07040'
    elif si >= 0.05: return '#c04060'
    else: return '#6b3fa0'

def si_to_tier_label(si):
    if si >= 0.75: return 'Safe / Low Risk'
    elif si >= 0.50: return 'Moderate Stress'
    elif si >= 0.25: return 'High / Critical Risk'
    elif si >= 0.05: return 'Uninhabitable Zone'
    else: return 'Extinction Zone'

# Add continent aggregates for missing years (2000-2020 and 2205-2500)
# For simplicity, extrapolate from the continental summary
for y in all_years:
    if y not in continental_summary:
        continental_summary[y] = {}
        for c in ['Africa', 'Asia', 'Europe', 'Middle East', 'North America', 'Oceania', 'South America']:
            # Find nearest years
            existing_years = sorted([yy for yy in continental_summary if c in continental_summary[yy]])
            if existing_years:
                if y < existing_years[0]:
                    ref = continental_summary[existing_years[0]][c]
                    continental_summary[y][c] = {k: v for k, v in ref.items()}
                else:
                    ref = continental_summary[existing_years[-1]][c]
                    continental_summary[y][c] = {k: v for k, v in ref.items()}

# ── Compute global aggregates per year ─────────────────────
global_agg = {}
for y in all_years:
    total_pop = 0
    total_mort = 0
    total_uninhab = 0
    si_sum = 0
    count = 0
    for region, data in ts.items():
        if y in data:
            d = data[y]
            total_pop += d['Population (M)']
            total_mort += d['Cumul. Excess Mortality (M)']
            si_sum += d['Survivability Index (SI)']
            count += 1
            if d['Habitability Status'] in ('Uninhabitable Zone',):
                total_uninhab += 1
    global_agg[y] = {
        'totalPop': round(total_pop, 1),
        'totalMortality': round(total_mort, 1),
        'avgSI': round(si_sum / max(1, count), 3),
        'pctUninhabitable': round(total_uninhab / max(1, count) * 100, 1),
        'uninhabitableCount': total_uninhab,
        'totalRegions': count
    }

# ── First uninhabitable year per region ────────────────────
first_uninhabitable = {}
for region, data in ts.items():
    for y in sorted(data.keys()):
        if data[y]['Habitability Status'] in ('Uninhabitable Zone',):
            first_uninhabitable[region] = y
            break

# ── Write outputs ──────────────────────────────────────────
output['tier_defs'] = tier_defs
output['continental'] = {str(y): v for y, v in continental_summary.items()}
output['global'] = {str(y): v for y, v in global_agg.items()}
output['first_uninhabitable'] = first_uninhabitable
output['region_meta'] = region_meta

import os

with open(OUT_JSON, 'w') as f:
    json.dump(output, f)

fsize = os.path.getsize(OUT_JSON)
print(f"✅ Simulation engine complete")
print(f"   Years: {all_years[0]}–{all_years[-1]} ({len(all_years)} steps)")
print(f"   Regions: {len(region_meta)}")
print(f"   Output: {OUT_JSON} ({fsize//1024} KB)")

# Print key milestones
print("\n📊 Key Milestones:")
print(f"{'Year':<8} {'Avg SI':<10} {'Pop (B)':<12} {'% Uninhabitable':<18} {'Cumul Deaths (B)':<18}")
for y in [2000, 2025, 2050, 2075, 2100, 2150, 2200, 2300, 2400, 2500]:
    g = global_agg.get(y, {})
    if g:
        print(f"{y:<8} {g['avgSI']:<10.3f} {g['totalPop']/1000:<12.2f} {g['pctUninhabitable']:<18.1f} {g['totalMortality']/1000:<18.2f}")

print("\n🔥 First Uninhabitable (top 10):")
for region, year in sorted(first_uninhabitable.items(), key=lambda x: x[1])[:10]:
    print(f"   {year}: {region} ({region_meta[region]['country']})")

import os
