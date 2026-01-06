#!/usr/bin/env python3
"""
Generate detailed test results report from test_data.txt

Usage: python3 generate_report.py [input_file] [output_file]
Defaults: test_data.txt -> test_results.md
"""

import math
import re
import sys
from pathlib import Path

# Constants
WIDTH = 16384
HEIGHT = 8192
EQUATOR_Y = 3340
EARTH_RADIUS_KM = 6371.0

R_MAP = (WIDTH * math.sqrt(2)) / (2 * math.pi)
Y_FACTOR = R_MAP * (1 + math.sqrt(2) / 2)

# Terrain cost factors
BASE_COSTS = {
    "flatland": 0.40,
    "coastal_ocean": 0.40,
    "inland_sea": 0.40,
    "plateau": 0.45,
    "hills": 0.50,
    "wetlands": 0.50,
    "mountains": 0.60,
    "lakes": 0.20,
    "wetlands_wasteland": 0.50,
    "narrows": 0.40,
}

# Water types - transitions to/from these use 0.20 factor
WATER_TYPES = {'lakes', 'coastal_ocean', 'inland_sea', 'narrows'}

VEG_COSTS = {
    "sparse": 0.00,
    "grasslands": 0.00,
    "desert": 0.00,
    "farmland": 0.02,
    "woods": 0.05,
    "forest": 0.10,
    "jungle": 0.10,
}

def load_terrain_data(templates_path):
    """Load terrain data from location_templates.txt"""
    terrain_db = {}

    try:
        with open(templates_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Parse: location_name = { topography = X vegetation = Y ... }
                match = re.match(r'^(\w+)\s*=\s*\{(.+)\}', line)
                if match:
                    name = match.group(1)
                    props = match.group(2)

                    topo_match = re.search(r'topography\s*=\s*(\w+)', props)
                    veg_match = re.search(r'vegetation\s*=\s*(\w+)', props)

                    terrain_db[name] = {
                        'topography': topo_match.group(1) if topo_match else 'unknown',
                        'vegetation': veg_match.group(1) if veg_match else '-',
                    }
    except FileNotFoundError:
        print(f"Warning: Could not find {templates_path}")

    return terrain_db

def get_terrain_cost(terrain_db, location):
    """Get terrain cost for a location"""
    if location not in terrain_db:
        return None, "unknown", "-"

    info = terrain_db[location]
    topo = info['topography']
    veg = info['vegetation']

    base = BASE_COSTS.get(topo, 0.40)
    veg_mod = VEG_COSTS.get(veg, 0.00)

    # Lakes override
    if topo == 'lakes':
        return 0.20, topo, veg

    return base + veg_mod, topo, veg

def pixel_distance(x1, y1, x2, y2):
    """Calculate Euclidean pixel distance"""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def pixel_to_latlon(x, y):
    """Convert pixel coordinates to lat/lon (radians)"""
    x_proj = x - (WIDTH / 2)
    lon = (x_proj * math.sqrt(2)) / R_MAP
    y_proj = y - EQUATOR_Y
    lat = 2 * math.atan(y_proj / Y_FACTOR)
    return lat, lon

def great_circle_km(lat1, lon1, lat2, lon2):
    """Calculate great circle distance in km"""
    dLon = lon2 - lon1
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2) * math.sin(dLon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return EARTH_RADIUS_KM * c

def parse_test_data(input_path):
    """Parse test_data.txt format"""
    tests = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 7:
                print(f"Warning: Skipping malformed line {line_num}: {line}")
                continue

            try:
                tests.append({
                    'line_num': line_num,  # Track source line number
                    'start': parts[0],
                    'start_x': int(parts[1]),
                    'start_y': int(parts[2]),
                    'end': parts[3],
                    'end_x': int(parts[4]),
                    'end_y': int(parts[5]),
                    'actual': float(parts[6]),
                    'notes': parts[7] if len(parts) > 7 else '',
                })
            except ValueError as e:
                print(f"Warning: Error parsing line {line_num}: {e}")

    return tests

def generate_report(tests, terrain_db, output_path):
    """Generate the markdown report"""

    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("# EU5 Market Access Distance Analysis\n\n")
        f.write("*Auto-generated from test_data.txt*\n\n")

        f.write("## Map Properties\n")
        f.write("- Dimensions: 16384 x 8192\n")
        f.write("- Equator Y: 3340\n")
        f.write("- Projection: Gall Stereographic\n\n")

        f.write("## Terrain Cost Factors\n\n")
        f.write("**Land Types:**\n")
        f.write("| Topography | Base | Vegetation | Modifier |\n")
        f.write("|------------|------|------------|----------|\n")
        f.write("| flatland   | 0.40 | sparse     | +0.00    |\n")
        f.write("| plateau    | 0.45 | grasslands | +0.00    |\n")
        f.write("| hills      | 0.50 | farmland   | +0.02    |\n")
        f.write("| wetlands   | 0.50 | woods      | +0.05    |\n")
        f.write("| mountains  | 0.60 | forest     | +0.10    |\n\n")
        f.write("**Water Types (always 0.20 for any transition involving water):**\n")
        f.write("- lakes, coastal_ocean, inland_sea, narrows\n\n")
        f.write("---\n\n")
        f.write("## Test Cases\n\n")

        matches = []
        outliers = []

        for test in tests:
            line_num = test['line_num']

            # Calculate values
            px_dist = pixel_distance(test['start_x'], test['start_y'],
                                     test['end_x'], test['end_y'])

            cost_start, topo_start, veg_start = get_terrain_cost(terrain_db, test['start'])
            cost_end, topo_end, veg_end = get_terrain_cost(terrain_db, test['end'])

            # Handle unknown terrain
            if cost_start is None:
                cost_start = 0.40
                topo_start = "unknown"
                veg_start = "-"
            if cost_end is None:
                cost_end = 0.40
                topo_end = "unknown"
                veg_end = "-"

            # Water transition detection
            is_water_start = topo_start in WATER_TYPES
            is_water_end = topo_end in WATER_TYPES

            # Water↔Land transitions use 0.20 factor (but water↔water uses normal costs)
            if is_water_start != is_water_end:
                avg_cost = 0.20
                expected = px_dist * 0.20
                cost_note = "water↔land 0.20"
            else:
                avg_cost = (cost_start + cost_end) / 2
                expected = px_dist * avg_cost
                cost_note = f"avg {avg_cost:.2f}"

            error = test['actual'] - expected

            # Calculate additional metrics
            dx = abs(test['end_x'] - test['start_x'])
            dy = abs(test['end_y'] - test['start_y'])

            lat1, lon1 = pixel_to_latlon(test['start_x'], test['start_y'])
            lat2, lon2 = pixel_to_latlon(test['end_x'], test['end_y'])
            avg_lat = math.degrees((lat1 + lat2) / 2)
            gc_km = great_circle_km(lat1, lon1, lat2, lon2)

            # Determine status
            is_outlier = abs(error) > 1.0
            if is_outlier:
                status = "**OUTLIER**"
                outliers.append((line_num, test['start'], test['end'], error, test['notes']))
            else:
                status = "**MATCH**"
                matches.append(line_num)

            # Write test case
            notes_str = f" ({test['notes']})" if test['notes'] else ""
            f.write(f"### Line {line_num}: {test['start']} → {test['end']}{notes_str}\n")
            f.write("| Location | X | Y | Topography | Vegetation | Cost |\n")
            f.write("|----------|---|---|------------|------------|------|\n")
            f.write(f"| {test['start']} | {test['start_x']} | {test['start_y']} | {topo_start} | {veg_start} | {cost_start:.2f} |\n")
            f.write(f"| {test['end']} | {test['end_x']} | {test['end_y']} | {topo_end} | {veg_end} | {cost_end:.2f} |\n\n")

            f.write(f"- Pixel Distance: {px_dist:.2f}\n")
            f.write(f"- In-Game Distance: {test['actual']:.2f}\n")
            f.write(f"- Expected ({cost_note}): {expected:.2f}\n")

            if is_outlier:
                ratio = test['actual'] / px_dist
                gc_pred = gc_km / 8 + 5
                f.write(f"- Error: {error:.2f}\n")
                f.write(f"- Ratio (actual/px): {ratio:.4f}\n")
                f.write(f"- Great Circle: {gc_km:.1f} km (gc/8+5 = {gc_pred:.2f})\n")
                f.write(f"- Movement: dx={dx}, dy={dy}, lat={avg_lat:.1f}°\n")

            f.write(f"- Status: {status}\n\n")
            f.write("---\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write(f"**Working Formula ({len(matches)}/{len(tests)} cases):**\n")
        f.write("```\n")
        f.write("If either location is water (lakes, coastal_ocean, inland_sea, narrows):\n")
        f.write("    In-Game Distance = Pixel Distance × 0.20\n")
        f.write("Else:\n")
        f.write("    In-Game Distance = Pixel Distance × Average(Terrain Cost₁, Terrain Cost₂)\n")
        f.write("```\n\n")

        if outliers:
            f.write(f"**Outliers ({len(outliers)} cases):**\n")
            f.write("| Line | Route | Error | Notes |\n")
            f.write("|------|-------|-------|-------|\n")
            for line_num, start, end, err, notes in outliers:
                f.write(f"| {line_num} | {start}→{end} | {err:.2f} | {notes} |\n")

    print(f"Generated report: {output_path}")
    print(f"  - {len(matches)} matches")
    print(f"  - {len(outliers)} outliers")

def main():
    # Paths
    script_dir = Path(__file__).parent
    input_path = script_dir / "test_data.txt"
    output_path = script_dir / "test_results.md"
    templates_path = script_dir / "map_data" / "location_templates.txt"

    # Override from command line
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])

    # Load terrain database
    print(f"Loading terrain data from {templates_path}...")
    terrain_db = load_terrain_data(templates_path)
    print(f"  Loaded {len(terrain_db)} locations")

    # Parse test data
    print(f"Parsing test data from {input_path}...")
    tests = parse_test_data(input_path)
    print(f"  Found {len(tests)} test cases")

    # Generate report
    print(f"Generating report...")
    generate_report(tests, terrain_db, output_path)

if __name__ == "__main__":
    main()
