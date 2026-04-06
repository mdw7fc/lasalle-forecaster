#!/usr/bin/env python3
"""
LaSalle Farm Water Balance Dashboard — Updater
================================================
Run this script to:
  1. Pull latest observed weather (temp, precip, ET0) from Open-Meteo
  2. Pull tiered multi-model ensemble forecasts:
     - Days 1-15:  ECMWF IFS ensemble (51 members, highest skill)
     - Days 15-46: EC46 / GFS 0.5° ensemble (sub-seasonal bridge)
     - Months 2-7: SEAS5 via ECMWF Seamless (seasonal baseline)
  3. Snapshot the current forecast for retrospective skill tracking
  4. Compute forecast skill metrics against actuals
  5. Rebuild dashboard_data.json (HTML loads it via fetch)

Usage:
    python update_dashboard.py

Requirements:
    pip install numpy
"""

import json
import math
import os
import sys
import numpy as np
from datetime import datetime, date, timedelta
from collections import defaultdict
from urllib.request import urlopen, Request
from urllib.error import URLError

# ── CONFIGURATION ──
LAT = 40.38
LON = -104.69
LOCATION_NAME = "LaSalle, CO"
DATA_FILE = "dashboard_data.json"
HISTORICAL_START = "1991-01-01"   # 30-year baseline (1991-2020) + recent years
SEASON_MONTHS = (4, 10)           # April through October

# ── SOIL PARAMETERS ──
AWC_INCHES = 7.0    # Available water capacity for LaSalle silt loam (~2ft root zone)
INITIAL_SW = 4.0    # Assumed soil water at season start (spring recharge)

# ── GDD PARAMETERS ──
GDD_BASE_CORN = 50     # Base temp (F) for corn GDD
GDD_BASE_SOYBEAN = 50  # Base temp (F) for soybean GDD
GDD_BASE_WHEAT = 32    # Base temp (F) for winter wheat GDD

# Corn Kc tied to cumulative GDD from planting (DOY 121)
# Thresholds: 0-200 GDD initial, 200-800 dev, 800-1800 mid, 1800-2600 late
CORN_KC_GDD = [(0, 0.30), (200, 0.30), (800, 1.20), (1800, 1.20), (2600, 0.60)]

# Soybean Kc tied to cumulative GDD from planting (DOY 135)
SOYBEAN_KC_GDD = [(0, 0.40), (150, 0.40), (600, 1.15), (1400, 1.15), (2200, 0.50)]

# Winter wheat Kc tied to cumulative spring GDD from DOY 60 (March 1)
WHEAT_KC_GDD = [(0, 0.40), (300, 0.40), (700, 1.15), (1300, 1.15), (1700, 0.25), (2000, 0.0)]

CROP_GDD_CONFIG = {
    'corn':    {'base_f': GDD_BASE_CORN,    'plant_doy': 121, 'harvest_doy': 274, 'kc_gdd': CORN_KC_GDD},
    'soybean': {'base_f': GDD_BASE_SOYBEAN, 'plant_doy': 135, 'harvest_doy': 274, 'kc_gdd': SOYBEAN_KC_GDD},
    'wheat':   {'base_f': GDD_BASE_WHEAT,   'plant_doy': 60,  'harvest_doy': 200, 'kc_gdd': WHEAT_KC_GDD},
}


def fetch_json(url):
    """Fetch JSON from URL with basic error handling."""
    try:
        req = Request(url, headers={'User-Agent': 'FarmDashboard/1.0'})
        with urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except URLError as e:
        print(f"  ERROR fetching {url[:80]}...: {e}")
        return None


def hargreaves_samani_et0(tmax_c, tmin_c, lat_rad, doy):
    """Estimate reference ET0 (mm/day) using Hargreaves-Samani equation.

    ET0 = 0.0023 * (Tmean + 17.8) * (Tmax - Tmin)^0.5 * Ra
    where Ra = extraterrestrial radiation (mm/day equivalent).
    """
    if tmax_c is None or tmin_c is None:
        return None
    tmean = (tmax_c + tmin_c) / 2.0
    td = max(tmax_c - tmin_c, 0)

    # Solar declination and day length for extraterrestrial radiation
    dr = 1 + 0.033 * math.cos(2 * math.pi * doy / 365)
    delta = 0.4093 * math.sin(2 * math.pi * doy / 365 - 1.405)
    ws = math.acos(max(-1, min(1, -math.tan(lat_rad) * math.tan(delta))))
    ra = (24 * 60 / math.pi) * 0.0820 * dr * (
        ws * math.sin(lat_rad) * math.sin(delta) +
        math.cos(lat_rad) * math.cos(delta) * math.sin(ws)
    )
    # Convert Ra from MJ/m²/day to mm/day equivalent (divide by 2.45)
    ra_mm = ra / 2.45

    et0 = 0.0023 * (tmean + 17.8) * math.sqrt(td) * ra_mm
    return max(et0, 0)


def scs_effective_precip(gross_precip_in, etc_in):
    """USDA SCS method for effective precipitation.

    More realistic than a flat 75% — accounts for intensity.
    For monthly P < 250mm, Peff = P * (1 - 0.2*P/125) approximately.
    Simplified daily version: tiered approach.
    """
    if gross_precip_in <= 0:
        return 0.0
    if gross_precip_in <= 0.20:
        return gross_precip_in * 0.95  # Light rain, nearly all effective
    elif gross_precip_in <= 0.50:
        return 0.20 * 0.95 + (gross_precip_in - 0.20) * 0.85
    elif gross_precip_in <= 1.0:
        return 0.20 * 0.95 + 0.30 * 0.85 + (gross_precip_in - 0.50) * 0.65
    else:
        return 0.20 * 0.95 + 0.30 * 0.85 + 0.50 * 0.65 + (gross_precip_in - 1.0) * 0.40


def interpolate_kc(gdd, kc_gdd_table):
    """Linearly interpolate Kc from a GDD-indexed table."""
    if gdd <= kc_gdd_table[0][0]:
        return kc_gdd_table[0][1]
    if gdd >= kc_gdd_table[-1][0]:
        return kc_gdd_table[-1][1]
    for i in range(len(kc_gdd_table) - 1):
        g0, k0 = kc_gdd_table[i]
        g1, k1 = kc_gdd_table[i + 1]
        if g0 <= gdd <= g1:
            frac = (gdd - g0) / (g1 - g0) if g1 > g0 else 0
            return k0 + frac * (k1 - k0)
    return kc_gdd_table[-1][1]


def compute_daily_gdd(tmax_f, tmin_f, base_f):
    """Compute growing degree days from Fahrenheit temps."""
    tmean = (tmax_f + tmin_f) / 2.0
    return max(tmean - base_f, 0)


def process_year(daily_data, year, use_fao_et0=True):
    """Compute cumulative ET0, precip, crop deficits, and soil water for a growing season.

    Uses Hargreaves-Samani ET0 when use_fao_et0=False (for forecast).
    Uses FAO PM ET0 from Open-Meteo when use_fao_et0=True (for historical).
    Includes soil water bucket model and GDD-based Kc.
    """
    lat_rad = math.radians(LAT)
    times = daily_data['time']
    tmax_raw = daily_data.get('temperature_2m_max', [None] * len(times))
    tmin_raw = daily_data.get('temperature_2m_min', [None] * len(times))
    et0_raw = daily_data.get('et0_fao_evapotranspiration', [None] * len(times))
    precip_raw = daily_data['precipitation_sum']

    result = {
        'dates': [], 'doy': [],
        'et0_daily': [], 'precip_daily': [],
        'et0_cum': [], 'precip_cum': [], 'deficit_cum': [],
        'tmax_f': [], 'tmin_f': [],
        'crop_et': {c: [] for c in CROP_GDD_CONFIG},
        'crop_et_cum': {c: [] for c in CROP_GDD_CONFIG},
        'crop_deficit_cum': {c: [] for c in CROP_GDD_CONFIG},
        'soil_water': {c: [] for c in CROP_GDD_CONFIG},
        'crop_gdd': {c: [] for c in CROP_GDD_CONFIG},
    }

    cum_et0 = cum_precip = cum_deficit = 0
    cum_crop_et = {c: 0 for c in CROP_GDD_CONFIG}
    cum_crop_def = {c: 0 for c in CROP_GDD_CONFIG}
    cum_gdd = {c: 0 for c in CROP_GDD_CONFIG}
    soil_water = {c: INITIAL_SW for c in CROP_GDD_CONFIG}

    for i, t in enumerate(times):
        d = datetime.strptime(t, '%Y-%m-%d')
        if d.year != year or d.month < SEASON_MONTHS[0] or d.month > SEASON_MONTHS[1]:
            continue

        doy = d.timetuple().tm_yday
        tmax_c = tmax_raw[i]
        tmin_c = tmin_raw[i]

        # ET0 in inches
        if use_fao_et0 and et0_raw[i] is not None:
            et0 = et0_raw[i] / 25.4  # mm to inches
        elif tmax_c is not None and tmin_c is not None:
            et0_mm = hargreaves_samani_et0(tmax_c, tmin_c, lat_rad, doy)
            et0 = (et0_mm or 0) / 25.4
        else:
            et0 = 0

        p = (precip_raw[i] or 0) / 25.4  # mm to inches

        # Temps in Fahrenheit for GDD and display
        tmax_f = (tmax_c * 9 / 5 + 32) if tmax_c is not None else None
        tmin_f = (tmin_c * 9 / 5 + 32) if tmin_c is not None else None

        cum_et0 += et0
        cum_precip += p
        cum_deficit += (p - et0)

        result['dates'].append(t)
        result['doy'].append(doy)
        result['et0_daily'].append(round(et0, 3))
        result['precip_daily'].append(round(p, 3))
        result['et0_cum'].append(round(cum_et0, 2))
        result['precip_cum'].append(round(cum_precip, 2))
        result['deficit_cum'].append(round(cum_deficit, 2))
        result['tmax_f'].append(round(tmax_f, 1) if tmax_f is not None else None)
        result['tmin_f'].append(round(tmin_f, 1) if tmin_f is not None else None)

        for crop, cfg in CROP_GDD_CONFIG.items():
            # Accumulate GDD from planting
            if doy >= cfg['plant_doy'] and doy <= cfg['harvest_doy']:
                if tmax_f is not None and tmin_f is not None:
                    cum_gdd[crop] += compute_daily_gdd(tmax_f, tmin_f, cfg['base_f'])
                kc = interpolate_kc(cum_gdd[crop], cfg['kc_gdd'])
            else:
                kc = 0.0

            crop_et_day = et0 * kc
            eff_p = scs_effective_precip(p, crop_et_day)

            cum_crop_et[crop] += crop_et_day
            cum_crop_def[crop] += (eff_p - crop_et_day)

            # Soil water bucket
            sw = soil_water[crop] + eff_p - crop_et_day
            sw = max(0, min(sw, AWC_INCHES))
            soil_water[crop] = sw

            result['crop_et'][crop].append(round(crop_et_day, 3))
            result['crop_et_cum'][crop].append(round(cum_crop_et[crop], 2))
            result['crop_deficit_cum'][crop].append(round(cum_crop_def[crop], 2))
            result['soil_water'][crop].append(round(sw, 2))
            result['crop_gdd'][crop].append(round(cum_gdd[crop], 0))

    return result


def compute_historical_envelope(historical_years):
    """Compute min/p25/median/p75/max envelope across all historical years."""
    all_doys = sorted(set().union(*[set(historical_years[yr]['doy']) for yr in historical_years]))

    env = {
        'doys': all_doys,
        'et0_cum': {'min': [], 'p25': [], 'median': [], 'p75': [], 'max': []},
        'precip_cum': {'min': [], 'p25': [], 'median': [], 'p75': [], 'max': []},
    }
    for crop in CROP_GDD_CONFIG:
        env[f'{crop}_deficit_cum'] = {'min': [], 'p25': [], 'median': [], 'p75': [], 'max': []}
        env[f'{crop}_soil_water'] = {'min': [], 'p25': [], 'median': [], 'p75': [], 'max': []}

    # Also compute temperature envelopes for weather tab
    env['tmax'] = {'median': [], 'p25': [], 'p75': []}
    env['tmin'] = {'median': [], 'p25': [], 'p75': []}

    for doy in all_doys:
        vals = {'et0_cum': [], 'precip_cum': [], 'tmax': [], 'tmin': []}
        crop_vals = {c: {'deficit': [], 'sw': []} for c in CROP_GDD_CONFIG}

        for yr in historical_years:
            yd = historical_years[yr]
            if doy in yd['doy']:
                idx = yd['doy'].index(doy)
                vals['et0_cum'].append(yd['et0_cum'][idx])
                vals['precip_cum'].append(yd['precip_cum'][idx])
                if yd.get('tmax_f') and yd['tmax_f'][idx] is not None:
                    vals['tmax'].append(yd['tmax_f'][idx])
                if yd.get('tmin_f') and yd['tmin_f'][idx] is not None:
                    vals['tmin'].append(yd['tmin_f'][idx])
                for c in CROP_GDD_CONFIG:
                    crop_vals[c]['deficit'].append(yd['crop_deficit_cum'][c][idx])
                    if 'soil_water' in yd:
                        crop_vals[c]['sw'].append(yd['soil_water'][c][idx])

        for key in ['et0_cum', 'precip_cum']:
            v = vals[key]
            if v:
                env[key]['min'].append(round(float(np.min(v)), 2))
                env[key]['p25'].append(round(float(np.percentile(v, 25)), 2))
                env[key]['median'].append(round(float(np.median(v)), 2))
                env[key]['p75'].append(round(float(np.percentile(v, 75)), 2))
                env[key]['max'].append(round(float(np.max(v)), 2))

        for temp_key in ['tmax', 'tmin']:
            v = vals[temp_key]
            if v:
                env[temp_key]['median'].append(round(float(np.median(v)), 1))
                env[temp_key]['p25'].append(round(float(np.percentile(v, 25)), 1))
                env[temp_key]['p75'].append(round(float(np.percentile(v, 75)), 1))

        for c in CROP_GDD_CONFIG:
            v = crop_vals[c]['deficit']
            if v:
                env[f'{c}_deficit_cum']['min'].append(round(float(np.min(v)), 2))
                env[f'{c}_deficit_cum']['p25'].append(round(float(np.percentile(v, 25)), 2))
                env[f'{c}_deficit_cum']['median'].append(round(float(np.median(v)), 2))
                env[f'{c}_deficit_cum']['p75'].append(round(float(np.percentile(v, 75)), 2))
                env[f'{c}_deficit_cum']['max'].append(round(float(np.max(v)), 2))

            v = crop_vals[c]['sw']
            if v:
                env[f'{c}_soil_water']['min'].append(round(float(np.min(v)), 2))
                env[f'{c}_soil_water']['p25'].append(round(float(np.percentile(v, 25)), 2))
                env[f'{c}_soil_water']['median'].append(round(float(np.median(v)), 2))
                env[f'{c}_soil_water']['p75'].append(round(float(np.percentile(v, 75)), 2))
                env[f'{c}_soil_water']['max'].append(round(float(np.max(v)), 2))

    return env


def extract_ensemble_daily(fc_data, max_members=51):
    """Extract per-member daily arrays from an ensemble API response.

    Returns dict: {date_str: [{'tmax_c', 'tmin_c', 'precip_mm'}, ...per member]}
    """
    fc_daily = fc_data['daily']
    fc_times = fc_daily['time']

    def get_keys(prefix):
        cols = [prefix] if prefix in fc_daily else []
        for i in range(1, max_members):
            k = f"{prefix}_member{i:02d}"
            if k in fc_daily:
                cols.append(k)
        return cols

    tmax_keys = get_keys('temperature_2m_max')
    tmin_keys = get_keys('temperature_2m_min')
    precip_keys = get_keys('precipitation_sum')
    n_members = len(tmax_keys)

    daily = {}
    for i, t in enumerate(fc_times):
        members = []
        for m in range(n_members):
            tmax_c = fc_daily[tmax_keys[m]][i] if m < len(tmax_keys) else None
            tmin_c = fc_daily[tmin_keys[m]][i] if m < len(tmin_keys) else None
            p_mm = fc_daily[precip_keys[m]][i] if m < len(precip_keys) else None
            members.append({'tmax_c': tmax_c, 'tmin_c': tmin_c, 'precip_mm': p_mm})
        daily[t] = members

    return daily, n_members


def blend_tiered_forecasts(ifs_data, gfs_data, seamless_data, today_str):
    """Blend tiered forecasts by lead time.

    Priority:
      Days 1-15:  IFS ensemble (highest resolution, highest skill)
      Days 15-46: Blend IFS (where available) + GFS + EC46/seamless
      Beyond 46:  SEAS5 seamless only

    Returns a unified daily dict: {date_str: [member dicts]}
    with a consistent number of members (resampled to 51).
    """
    today = datetime.strptime(today_str, '%Y-%m-%d').date()

    # Collect all dates across all sources
    all_dates = set()
    for src in [ifs_data, gfs_data, seamless_data]:
        if src:
            all_dates.update(src.keys())
    all_dates = sorted(all_dates)

    blended = {}
    source_map = {}  # Track which source was used for each date

    for date_str in all_dates:
        d = datetime.strptime(date_str, '%Y-%m-%d').date()
        lead_days = (d - today).days

        if lead_days <= 15 and ifs_data and date_str in ifs_data:
            # Tier 1: IFS ensemble (best skill for days 1-15)
            blended[date_str] = ifs_data[date_str]
            source_map[date_str] = 'IFS'
        elif lead_days <= 46:
            # Tier 2: Blend available sub-seasonal models
            sources = []
            if ifs_data and date_str in ifs_data:
                sources.append(('IFS', ifs_data[date_str]))
            if gfs_data and date_str in gfs_data:
                sources.append(('GFS', gfs_data[date_str]))
            if seamless_data and date_str in seamless_data:
                sources.append(('EC46', seamless_data[date_str]))

            if not sources:
                continue

            # Pool all members from available sources
            pooled = []
            for name, members in sources:
                pooled.extend(members)
            blended[date_str] = pooled
            source_map[date_str] = '+'.join(s[0] for s in sources)
        else:
            # Tier 3: Seasonal (SEAS5 via seamless)
            if seamless_data and date_str in seamless_data:
                blended[date_str] = seamless_data[date_str]
                source_map[date_str] = 'SEAS5'

    return blended, source_map


def process_blended_forecast(blended_daily, source_map, hist_daily):
    """Process blended multi-model forecast into cumulative stats."""
    lat_rad = math.radians(LAT)

    # Build DOY-level normals from historical for fallback
    doy_et0 = defaultdict(list)
    for i, t in enumerate(hist_daily['time']):
        d = datetime.strptime(t, '%Y-%m-%d')
        if 4 <= d.month <= 10:
            doy = d.timetuple().tm_yday
            et0 = hist_daily.get('et0_fao_evapotranspiration', [None] * len(hist_daily['time']))[i]
            if et0 is not None:
                doy_et0[doy].append(et0 / 25.4)

    # Get sorted dates in growing season
    sorted_dates = sorted(blended_daily.keys())
    season_dates = []
    for t in sorted_dates:
        d = datetime.strptime(t, '%Y-%m-%d')
        if 4 <= d.month <= 10:
            season_dates.append(t)

    if not season_dates:
        return None, None

    # Determine max members across all dates
    max_members = max(len(blended_daily[t]) for t in season_dates)
    # Normalize to consistent member count by resampling
    target_members = min(max_members, 51)

    # Process each "virtual member" — for dates with fewer members, resample
    member_curves = []
    for mem_idx in range(target_members):
        cum = {'et0': 0, 'precip': 0}
        cum_crop = {c: {'et': 0, 'def': 0} for c in CROP_GDD_CONFIG}
        cum_gdd = {c: 0 for c in CROP_GDD_CONFIG}
        soil_water = {c: INITIAL_SW for c in CROP_GDD_CONFIG}
        pts = []

        for t in season_dates:
            d = datetime.strptime(t, '%Y-%m-%d')
            doy = d.timetuple().tm_yday
            members = blended_daily[t]
            n = len(members)

            # Map this member index to available members (wrap around)
            actual_idx = mem_idx % n
            m = members[actual_idx]

            tmax_c = m['tmax_c']
            tmin_c = m['tmin_c']
            p_mm = m['precip_mm']

            if tmax_c is None:
                continue

            # Hargreaves-Samani ET0
            if tmin_c is not None:
                et0_mm = hargreaves_samani_et0(tmax_c, tmin_c, lat_rad, doy)
                et0_in = (et0_mm or 0) / 25.4
            else:
                normal_et0 = np.mean(doy_et0.get(doy, [0.15]))
                et0_in = normal_et0  # fallback to climatological ET0
                et0_in = max(et0_in, 0)

            p_in = (p_mm / 25.4) if p_mm else 0
            tmax_f = tmax_c * 9 / 5 + 32
            tmin_f = (tmin_c * 9 / 5 + 32) if tmin_c is not None else tmax_f - 20

            cum['et0'] += et0_in
            cum['precip'] += p_in

            crops = {}
            for c, cfg in CROP_GDD_CONFIG.items():
                if cfg['plant_doy'] <= doy <= cfg['harvest_doy']:
                    cum_gdd[c] += compute_daily_gdd(tmax_f, tmin_f, cfg['base_f'])
                    kc = interpolate_kc(cum_gdd[c], cfg['kc_gdd'])
                else:
                    kc = 0.0

                cet = et0_in * kc
                eff_p = scs_effective_precip(p_in, cet)
                cum_crop[c]['et'] += cet
                cum_crop[c]['def'] += (eff_p - cet)

                sw = soil_water[c] + eff_p - cet
                sw = max(0, min(sw, AWC_INCHES))
                soil_water[c] = sw

                crops[c] = {
                    'cum_et': round(cum_crop[c]['et'], 2),
                    'cum_def': round(cum_crop[c]['def'], 2),
                    'sw': round(sw, 2),
                    'gdd': round(cum_gdd[c], 0),
                }

            pts.append({
                'date': t, 'doy': doy,
                'et0_cum': round(cum['et0'], 2),
                'precip_cum': round(cum['precip'], 2),
                'tmax_f': round(tmax_f, 1),
                'tmin_f': round(tmin_f, 1),
                'crops': crops,
                'source': source_map.get(t, ''),
            })
        member_curves.append(pts)

    if not member_curves or not member_curves[0]:
        return None, None

    # Aggregate stats
    fc_dates = [p['date'] for p in member_curves[0]]
    fc_doys = [p['doy'] for p in member_curves[0]]
    fc_sources = [p['source'] for p in member_curves[0]]

    stats = {
        'dates': fc_dates, 'doys': fc_doys, 'sources': fc_sources,
        'et0_cum': {'mean': [], 'p10': [], 'p90': []},
        'precip_cum': {'mean': [], 'p10': [], 'p90': []},
    }
    for c in CROP_GDD_CONFIG:
        stats[f'{c}_et_cum'] = {'mean': [], 'p10': [], 'p90': []}
        stats[f'{c}_deficit_cum'] = {'mean': [], 'p10': [], 'p90': []}
        stats[f'{c}_soil_water'] = {'mean': [], 'p10': [], 'p90': []}

    wx = {
        'dates': fc_dates, 'doys': fc_doys,
        'tmax': {'mean': [], 'p10': [], 'p25': [], 'p75': [], 'p90': []},
        'tmin': {'mean': [], 'p10': [], 'p25': [], 'p75': [], 'p90': []},
        'precip_cum': {'mean': [], 'p10': [], 'p25': [], 'p75': [], 'p90': []},
    }

    for step in range(len(fc_dates)):
        valid = [m for m in range(target_members) if step < len(member_curves[m])]
        if not valid:
            continue

        et0s = [member_curves[m][step]['et0_cum'] for m in valid]
        ps = [member_curves[m][step]['precip_cum'] for m in valid]
        tmaxs = [member_curves[m][step]['tmax_f'] for m in valid]
        tmins = [member_curves[m][step]['tmin_f'] for m in valid]

        for key, vals in [('et0_cum', et0s), ('precip_cum', ps)]:
            stats[key]['mean'].append(round(float(np.mean(vals)), 2))
            stats[key]['p10'].append(round(float(np.percentile(vals, 10)), 2))
            stats[key]['p90'].append(round(float(np.percentile(vals, 90)), 2))

        for c in CROP_GDD_CONFIG:
            cets = [member_curves[m][step]['crops'][c]['cum_et'] for m in valid]
            cdefs = [member_curves[m][step]['crops'][c]['cum_def'] for m in valid]
            sws = [member_curves[m][step]['crops'][c]['sw'] for m in valid]
            for arr_key, arr_vals in [(f'{c}_et_cum', cets), (f'{c}_deficit_cum', cdefs), (f'{c}_soil_water', sws)]:
                stats[arr_key]['mean'].append(round(float(np.mean(arr_vals)), 2))
                stats[arr_key]['p10'].append(round(float(np.percentile(arr_vals, 10)), 2))
                stats[arr_key]['p90'].append(round(float(np.percentile(arr_vals, 90)), 2))

        for key, vals in [('tmax', tmaxs), ('tmin', tmins)]:
            wx[key]['mean'].append(round(float(np.mean(vals)), 1))
            wx[key]['p10'].append(round(float(np.percentile(vals, 10)), 1))
            wx[key]['p25'].append(round(float(np.percentile(vals, 25)), 1))
            wx[key]['p75'].append(round(float(np.percentile(vals, 75)), 1))
            wx[key]['p90'].append(round(float(np.percentile(vals, 90)), 1))

        pcums = [member_curves[m][step]['precip_cum'] for m in valid]
        wx['precip_cum']['mean'].append(round(float(np.mean(pcums)), 2))
        wx['precip_cum']['p10'].append(round(float(np.percentile(pcums, 10)), 2))
        wx['precip_cum']['p25'].append(round(float(np.percentile(pcums, 25)), 2))
        wx['precip_cum']['p75'].append(round(float(np.percentile(pcums, 75)), 2))
        wx['precip_cum']['p90'].append(round(float(np.percentile(pcums, 90)), 2))

    return stats, wx


def compute_monthly_wx_summary(wx_forecast, hist_envelope):
    """Compute monthly weather summary: forecast mean vs 1991-2020 normals."""
    # 1991-2020 normals for LaSalle area (NOAA)
    normals = {
        'Apr': {'tmax': 60, 'tmin': 30, 'precip': 1.83},
        'May': {'tmax': 70, 'tmin': 40, 'precip': 2.65},
        'Jun': {'tmax': 82, 'tmin': 50, 'precip': 1.78},
        'Jul': {'tmax': 90, 'tmin': 57, 'precip': 1.61},
        'Aug': {'tmax': 87, 'tmin': 55, 'precip': 1.56},
        'Sep': {'tmax': 79, 'tmin': 45, 'precip': 1.13},
    }
    month_doy_ranges = {
        'Apr': (91, 120), 'May': (121, 151), 'Jun': (152, 181),
        'Jul': (182, 212), 'Aug': (213, 243), 'Sep': (244, 273),
    }

    summary = []
    doys = wx_forecast['doys']

    for month, (doy_start, doy_end) in month_doy_ranges.items():
        indices = [i for i, d in enumerate(doys) if doy_start <= d <= doy_end]
        if not indices:
            continue
        n = normals[month]
        avg_tmax = np.mean([wx_forecast['tmax']['mean'][i] for i in indices])
        avg_tmin = np.mean([wx_forecast['tmin']['mean'][i] for i in indices])

        # Precip for this month: difference of cumulative at end vs start
        pcum_end = wx_forecast['precip_cum']['mean'][indices[-1]]
        pcum_start = wx_forecast['precip_cum']['mean'][indices[0] - 1] if indices[0] > 0 else 0
        precip_fc = pcum_end - pcum_start

        summary.append({
            'month': month,
            'tmax_fc': round(float(avg_tmax), 1),
            'tmax_norm': n['tmax'],
            'tmax_anom': round(float(avg_tmax - n['tmax']), 1),
            'tmin_fc': round(float(avg_tmin), 1),
            'tmin_norm': n['tmin'],
            'tmin_anom': round(float(avg_tmin - n['tmin']), 1),
            'precip_fc': round(float(precip_fc), 2),
            'precip_norm': n['precip'],
            'precip_anom': round(float(precip_fc - n['precip']), 2),
        })

    return summary


def compute_forecast_skill(forecast_snapshots, observed_2026):
    """Compute forecast skill metrics: bias, MAE, and RMSE for each snapshot vs actuals."""
    if not observed_2026 or not observed_2026.get('doy') or not forecast_snapshots:
        return []

    obs_doys = observed_2026['doy']
    if not obs_doys:
        return []

    skills = []
    for snap in forecast_snapshots:
        fc = snap.get('forecast', {})
        fc_doys = fc.get('doys', [])
        if not fc_doys:
            continue

        snap_skill = {'date': snap['date'], 'label': snap.get('label', ''), 'crops': {}}

        for crop in CROP_GDD_CONFIG:
            fc_def_key = f'{crop}_deficit_cum'
            fc_et_key = f'{crop}_et_cum'
            if fc_def_key not in fc or crop not in observed_2026.get('crop_deficit_cum', {}):
                continue

            obs_def = observed_2026['crop_deficit_cum'][crop]
            obs_et = observed_2026['crop_et_cum'][crop]
            fc_def_mean = fc[fc_def_key]['mean']
            fc_et_mean = fc[fc_et_key]['mean']

            # Match on common DOYs
            errors_def = []
            errors_et = []
            for j, odoy in enumerate(obs_doys):
                if odoy in fc_doys:
                    fi = fc_doys.index(odoy)
                    errors_def.append(fc_def_mean[fi] - obs_def[j])
                    errors_et.append(fc_et_mean[fi] - obs_et[j])

            if errors_def:
                snap_skill['crops'][crop] = {
                    'n_days': len(errors_def),
                    'deficit_bias': round(float(np.mean(errors_def)), 2),
                    'deficit_mae': round(float(np.mean(np.abs(errors_def))), 2),
                    'et_bias': round(float(np.mean(errors_et)), 2),
                    'et_mae': round(float(np.mean(np.abs(errors_et))), 2),
                    'last_fc_def': round(float(fc_def_mean[min(len(fc_def_mean) - 1, fc_doys.index(obs_doys[-1]) if obs_doys[-1] in fc_doys else len(fc_def_mean) - 1)]), 2),
                    'last_obs_def': round(float(obs_def[-1]), 2),
                }

        skills.append(snap_skill)

    return skills


def main():
    today = date.today().isoformat()
    print(f"=== LaSalle Farm Dashboard Update — {today} ===\n")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, DATA_FILE)

    # ── STEP 1: Load or initialize data store ──
    if os.path.exists(data_path):
        print("Loading existing dashboard data...")
        with open(data_path) as f:
            dashboard = json.load(f)
        print(f"  Found {len(dashboard.get('forecast_snapshots', []))} previous forecast snapshots")
    else:
        print("No existing data found. Will build from scratch.")
        dashboard = {
            'metadata': {}, 'forecast_snapshots': [], 'forecast_skill': [],
            'historical_years': {}, 'historical_envelope': {},
            'observed_2026': {}, 'current_forecast': {},
            'wx_forecast': {}, 'wx_hist_envelope': {},
            'obs_wx': {}, 'monthly_wx_summary': [],
        }

    # ── STEP 2: Fetch historical data (30 years: 1991-2025) ──
    print("\nFetching historical weather data (1991-2025)...")
    hist_end = "2025-12-31"
    hist_url = (f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={LAT}&longitude={LON}&start_date={HISTORICAL_START}&end_date={hist_end}"
                f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,et0_fao_evapotranspiration"
                f"&timezone=America/Denver")
    hist_data = fetch_json(hist_url)
    if not hist_data:
        print("  FAILED to fetch historical data. Using cached version.")
    else:
        print(f"  Got {len(hist_data['daily']['time'])} days")
        historical_years = {}
        for yr in range(1991, 2026):
            r = process_year(hist_data['daily'], yr, use_fao_et0=True)
            if len(r['dates']) > 100:
                historical_years[str(yr)] = r
        dashboard['historical_years'] = historical_years
        dashboard['historical_envelope'] = compute_historical_envelope(historical_years)
        print(f"  Processed {len(historical_years)} historical years")

    # ── STEP 3: Fetch 2026 observed data through today ──
    print(f"\nFetching 2026 observed data through {today}...")
    obs_url = (f"https://archive-api.open-meteo.com/v1/archive?"
               f"latitude={LAT}&longitude={LON}&start_date=2026-01-01&end_date={today}"
               f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,et0_fao_evapotranspiration"
               f"&timezone=America/Denver")
    obs_data = fetch_json(obs_url)
    if obs_data:
        obs_2026 = process_year(obs_data['daily'], 2026, use_fao_et0=True)
        dashboard['observed_2026'] = obs_2026
        n_obs = len(obs_2026['dates'])
        print(f"  Got {n_obs} growing-season days observed")
        if n_obs > 0:
            print(f"  YTD ET0: {obs_2026['et0_cum'][-1]:.1f}\", Precip: {obs_2026['precip_cum'][-1]:.1f}\"")

        # obs_wx for weather tab
        dashboard['obs_wx'] = {
            'dates': obs_2026['dates'],
            'doys': obs_2026['doy'],
            'tmax': obs_2026['tmax_f'],
            'tmin': obs_2026['tmin_f'],
            'precip_cum': obs_2026['precip_cum'],
        }

    # ── STEP 4: Fetch tiered multi-model forecasts ──
    # Tier 1: ECMWF IFS ensemble (days 1-15, 51 members, highest skill)
    print("\n── Tiered Forecast Fetch ──")
    print("Tier 1: Fetching ECMWF IFS ensemble (days 1-15)...")
    ifs_url = (f"https://ensemble-api.open-meteo.com/v1/ensemble?"
               f"latitude={LAT}&longitude={LON}"
               f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
               f"&models=ecmwf_ifs025"
               f"&timezone=America/Denver")
    ifs_data_raw = fetch_json(ifs_url)
    ifs_daily = None
    if ifs_data_raw:
        ifs_daily, ifs_n = extract_ensemble_daily(ifs_data_raw)
        print(f"  IFS: {len(ifs_daily)} days, {ifs_n} members")
    else:
        print("  IFS: not available (will fall back to other models)")

    # Tier 2: GFS 0.5° ensemble (days 1-35, 31 members, independent model)
    print("Tier 2: Fetching GFS 0.5° ensemble (days 1-35)...")
    gfs_url = (f"https://ensemble-api.open-meteo.com/v1/ensemble?"
               f"latitude={LAT}&longitude={LON}"
               f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
               f"&models=gfs05"
               f"&timezone=America/Denver")
    gfs_data_raw = fetch_json(gfs_url)
    gfs_daily = None
    if gfs_data_raw:
        gfs_daily, gfs_n = extract_ensemble_daily(gfs_data_raw)
        print(f"  GFS: {len(gfs_daily)} days, {gfs_n} members")
    else:
        print("  GFS: not available")

    # Tier 3: ECMWF Seamless (EC46 days 1-46 + SEAS5 months 2-7, 51 members)
    print("Tier 3: Fetching ECMWF Seamless (EC46 + SEAS5)...")
    seamless_url = (f"https://seasonal-api.open-meteo.com/v1/seasonal?"
                    f"latitude={LAT}&longitude={LON}"
                    f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
                    f"&timezone=America/Denver")
    seamless_data_raw = fetch_json(seamless_url)
    seamless_daily = None
    if seamless_data_raw:
        seamless_daily, seamless_n = extract_ensemble_daily(seamless_data_raw)
        print(f"  Seamless: {len(seamless_daily)} days, {seamless_n} members")
    else:
        print("  Seamless: not available")

    # Blend by lead time
    any_forecast = ifs_daily or gfs_daily or seamless_daily
    if any_forecast and hist_data:
        print("\nBlending forecasts by lead time...")
        blended, source_map = blend_tiered_forecasts(ifs_daily, gfs_daily, seamless_daily, today)

        # Count sources
        src_counts = defaultdict(int)
        for s in source_map.values():
            src_counts[s] += 1
        for src, count in sorted(src_counts.items()):
            print(f"  {src}: {count} days")

        result = process_blended_forecast(blended, source_map, hist_data['daily'])
        if result and result[0]:
            new_forecast, wx_forecast = result

            # Snapshot the previous forecast before replacing
            if dashboard.get('current_forecast') and dashboard['current_forecast'].get('dates'):
                prev_date = dashboard['metadata'].get('generated', 'unknown')
                existing_dates = [s['date'] for s in dashboard.get('forecast_snapshots', [])]
                if prev_date not in existing_dates:
                    dashboard['forecast_snapshots'].append({
                        'date': prev_date,
                        'label': f'Forecast as of {prev_date}',
                        'forecast': dashboard['current_forecast']
                    })
                    print(f"  Saved forecast snapshot from {prev_date}")

            dashboard['current_forecast'] = new_forecast
            dashboard['wx_forecast'] = wx_forecast
            dashboard['wx_hist_envelope'] = {
                'doys': dashboard['historical_envelope']['doys'],
                'tmax': dashboard['historical_envelope'].get('tmax', {}),
                'tmin': dashboard['historical_envelope'].get('tmin', {}),
            }
            dashboard['monthly_wx_summary'] = compute_monthly_wx_summary(
                wx_forecast, dashboard['historical_envelope']
            )
            print(f"  Blended forecast: {len(new_forecast['dates'])} days")

    # ── STEP 5: Compute forecast skill metrics ──
    print("\nComputing forecast skill metrics...")
    dashboard['forecast_skill'] = compute_forecast_skill(
        dashboard.get('forecast_snapshots', []),
        dashboard.get('observed_2026', {})
    )
    print(f"  Evaluated {len(dashboard['forecast_skill'])} snapshots")

    # ── STEP 6: Update metadata ──
    all_years = sorted([int(y) for y in dashboard.get('historical_years', {}).keys()])
    dashboard['metadata'] = {
        'location': f"{LOCATION_NAME} ({LAT}N, {LON}W)",
        'generated': today,
        'forecast_source': 'Tiered: IFS (d1-15) + GFS/EC46 (d15-46) + SEAS5 (m2-7)',
        'historical_source': 'Open-Meteo ERA5 reanalysis',
        'et0_method': 'FAO Penman-Monteith (historical), Hargreaves-Samani (forecast)',
        'precip_method': 'USDA SCS effective precipitation',
        'soil_model': f'Single-layer bucket, AWC={AWC_INCHES}" (silt loam)',
        'phenology': 'GDD-based crop coefficients (FAO-56 adapted)',
        'crops': list(CROP_GDD_CONFIG.keys()),
        'years': all_years,
        'baseline_years': [y for y in all_years if 1991 <= y <= 2020],
    }

    # ── STEP 7: Save data ──
    print(f"\nSaving data to {data_path}...")
    with open(data_path, 'w') as f:
        json.dump(dashboard, f)
    print(f"  Data file: {os.path.getsize(data_path) / 1024:.0f} KB")

    # ── SUMMARY ──
    print(f"\n{'=' * 60}")
    print(f"Update complete.")
    print(f"  Historical years: {len(all_years)} ({all_years[0]}-{all_years[-1]})")
    print(f"  Forecast snapshots: {len(dashboard.get('forecast_snapshots', []))}")
    print(f"  Forecast: IFS (d1-15) + GFS/EC46 (d15-46) + SEAS5 (m2-7)")
    print(f"  Model: Hargreaves-Samani ET0 | SCS eff. precip | Soil bucket (AWC={AWC_INCHES}\")")
    print(f"  Open farm_water_balance_dashboard.html in a browser to view.")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
