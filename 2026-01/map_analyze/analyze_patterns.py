#!/usr/bin/env python3
"""Analyze patterns in outliers vs matches"""

import math
import re
from pathlib import Path

WIDTH = 16384
EQUATOR_Y = 3340
R_MAP = (WIDTH * math.sqrt(2)) / (2 * math.pi)
Y_FACTOR = R_MAP * (1 + math.sqrt(2) / 2)

def get_lat(y):
    y_proj = y - EQUATOR_Y
    lat_rad = 2 * math.atan(y_proj / Y_FACTOR)
    return math.degrees(lat_rad)

# Parse directly from test_data.txt
def parse_test_data(path):
    tests = []
    with open(path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 7:
                try:
                    tests.append({
                        'line': line_num,
                        'start': parts[0],
                        'start_x': int(parts[1]),
                        'start_y': int(parts[2]),
                        'end': parts[3],
                        'end_x': int(parts[4]),
                        'end_y': int(parts[5]),
                        'actual': float(parts[6]),
                        'notes': parts[7] if len(parts) > 7 else '',
                    })
                except:
                    pass
    return tests

# Load terrain
def load_terrain(path):
    terrain = {}
    with open(path, 'r') as f:
        for line in f:
            match = re.match(r'^(\w+)\s*=\s*\{(.+)\}', line.strip())
            if match:
                name = match.group(1)
                props = match.group(2)
                topo = re.search(r'topography\s*=\s*(\w+)', props)
                terrain[name] = topo.group(1) if topo else 'unknown'
    return terrain

script_dir = Path(__file__).parent
tests = parse_test_data(script_dir / 'test_data.txt')
terrain = load_terrain(script_dir / 'map_data' / 'location_templates.txt')

# Calculate metrics for each test
for t in tests:
    t['dx'] = abs(t['end_x'] - t['start_x'])
    t['dy'] = abs(t['end_y'] - t['start_y'])
    t['px_dist'] = math.sqrt(t['dx']**2 + t['dy']**2)
    t['ratio'] = t['actual'] / t['px_dist'] if t['px_dist'] > 0 else 0
    t['avg_y'] = (t['start_y'] + t['end_y']) / 2
    t['topo1'] = terrain.get(t['start'], 'unknown')
    t['topo2'] = terrain.get(t['end'], 'unknown')

# Classify as match/outlier based on expected terrain formula
BASE_COSTS = {"flatland": 0.40, "coastal_ocean": 0.40, "inland_sea": 0.40,
              "plateau": 0.45, "hills": 0.50, "wetlands": 0.50, "lakes": 0.20,
              "narrows": 0.40, "mountains": 0.60}

for t in tests:
    c1 = BASE_COSTS.get(t['topo1'], 0.40)
    c2 = BASE_COSTS.get(t['topo2'], 0.40)
    if t['topo1'] == 'lakes' or t['topo2'] == 'lakes':
        expected = t['px_dist'] * 0.20
    else:
        expected = t['px_dist'] * (c1 + c2) / 2
    t['expected'] = expected
    t['error'] = t['actual'] - expected
    t['is_outlier'] = abs(t['error']) > 1.0

matches = [t for t in tests if not t['is_outlier']]
outliers = [t for t in tests if t['is_outlier']]

print("=" * 110)
print(f"PATTERN ANALYSIS: {len(tests)} tests, {len(matches)} matches, {len(outliers)} outliers")
print("=" * 110)

print("\n" + "=" * 110)
print("OUTLIERS DETAIL (sorted by Y coordinate)")
print("=" * 110)
print(f"\n{'Line':<5} {'Route':<35} {'Topo':<25} {'Y':<6} {'dx':<5} {'dy':<5} {'Ratio':<7}")
print("-" * 100)
for t in sorted(outliers, key=lambda x: -x['avg_y']):
    route = f"{t['start']}→{t['end']}"
    topo = f"{t['topo1']}→{t['topo2']}"
    print(f"{t['line']:<5} {route:<35} {topo:<25} {t['avg_y']:<6.0f} {t['dx']:<5} {t['dy']:<5} {t['ratio']:<7.4f}")

# Group outliers by pattern
print("\n" + "=" * 110)
print("OUTLIER PATTERNS")
print("=" * 110)

water_types = ['lakes', 'inland_sea', 'coastal_ocean', 'narrows']

print("\n--- Pattern A: Water↔Land Transitions (ratio ~0.20) ---")
for t in outliers:
    is_water1 = t['topo1'] in water_types
    is_water2 = t['topo2'] in water_types
    if is_water1 != is_water2 or t['topo1'] == 'narrows' or t['topo2'] == 'narrows':
        print(f"  Line {t['line']}: {t['topo1']}→{t['topo2']}, ratio={t['ratio']:.4f}")

print("\n--- Pattern B: High-Latitude E-W (deatnu area, Y>7700) ---")
for t in outliers:
    is_water1 = t['topo1'] in water_types
    is_water2 = t['topo2'] in water_types
    if t['avg_y'] > 7700 and not (is_water1 != is_water2):
        print(f"  Line {t['line']}: {t['start']}→{t['end']}, Y={t['avg_y']:.0f}, dx={t['dx']}, dy={t['dy']}, ratio={t['ratio']:.4f}")

print("\n--- Pattern C: Other/Unexplained ---")
for t in outliers:
    is_water1 = t['topo1'] in water_types
    is_water2 = t['topo2'] in water_types
    is_water_transition = (is_water1 != is_water2) or t['topo1'] == 'narrows' or t['topo2'] == 'narrows'
    is_high_lat = t['avg_y'] > 7700
    if not is_water_transition and not is_high_lat:
        print(f"  Line {t['line']}: {t['start']}→{t['end']}, Y={t['avg_y']:.0f}, dx={t['dx']}, dy={t['dy']}, ratio={t['ratio']:.4f}")

print("\n" + "=" * 110)
print("KEY COMPARISON: High-lat areas that WORK vs DON'T WORK")
print("=" * 110)

print("\n--- yarok area (Y~7850, North Russia) - ALL MATCH ---")
for t in tests:
    if t['start'] == 'yarok':
        print(f"  Line {t['line']}: →{t['end']}, dx={t['dx']}, dy={t['dy']}, ratio={t['ratio']:.4f}, {'OUTLIER' if t['is_outlier'] else 'MATCH'}")

print("\n--- deatnu area (Y~7800, Norway) - ALL OUTLIER ---")
for t in tests:
    if t['start'] == 'deatnu':
        print(f"  Line {t['line']}: →{t['end']}, dx={t['dx']}, dy={t['dy']}, ratio={t['ratio']:.4f}, {'OUTLIER' if t['is_outlier'] else 'MATCH'}")

print("\n--- korvala area (Y~7500, Finland) - ALL MATCH ---")
for t in tests:
    if t['start'] == 'korvala':
        print(f"  Line {t['line']}: →{t['end']}, dx={t['dx']}, dy={t['dy']}, ratio={t['ratio']:.4f}, {'OUTLIER' if t['is_outlier'] else 'MATCH'}")

print("\n" + "=" * 110)
print("HYPOTHESIS: Is there something special about deatnu's X position?")
print("=" * 110)
print(f"\n  deatnu X = 8698 (longitude region: Scandinavia)")
print(f"  yarok X = 13947 (longitude region: Siberia)")
print(f"  korvala X = 8883 (longitude region: Finland)")
print(f"\n  deatnu and korvala are at similar X (~8700-8900)")
print(f"  But deatnu fails and korvala works...")
print(f"\n  Key difference: deatnu Y=7801, korvala Y=7488")
print(f"  Y difference = 313 pixels (~3.5° latitude)")

# Check if there's a Y threshold
print("\n" + "=" * 110)
print("CHECKING Y THRESHOLD")
print("=" * 110)

high_y_tests = [t for t in tests if t['avg_y'] > 7400]
print(f"\nAll tests with Y > 7400, sorted by Y:")
print(f"{'Line':<5} {'Start':<15} {'Y':<6} {'dx':<5} {'Status':<8}")
print("-" * 50)
for t in sorted(high_y_tests, key=lambda x: x['avg_y']):
    status = 'OUTLIER' if t['is_outlier'] else 'match'
    print(f"{t['line']:<5} {t['start']:<15} {t['avg_y']:<6.0f} {t['dx']:<5} {status:<8}")
