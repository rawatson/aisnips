# EU5 Market Access Distance Formula - Complete Analysis

*Reverse-engineered from Europa Universalis V (Patch 1.10)*

## Executive Summary

The EU5 market access system calculates base distances between locations using a simple formula based on **pixel distance** on the game's Gall Stereographic projection map, modified by terrain costs. However, this approach introduces significant **latitude-based distortion** that penalizes high-latitude regions (Scandinavia, Russia, Alaska, Patagonia) with market access distances far exceeding their actual geographic distances.

---

## The Formula

```
If exactly one location is water (lakes, coastal_ocean, inland_sea, narrows):
    In-Game Distance = Pixel Distance × 0.20

Otherwise:
    In-Game Distance = Pixel Distance × Average(Terrain Cost₁, Terrain Cost₂)
```

Where:
- **Pixel Distance** = √((x₂-x₁)² + (y₂-y₁)²) on the 16384×8192 game map
- **Terrain Cost** = Base topography cost + Vegetation modifier

### Terrain Costs

| Topography | Base Cost | | Vegetation | Modifier |
|------------|-----------|---|------------|----------|
| flatland   | 0.40      | | sparse     | +0.00    |
| plateau    | 0.45      | | grasslands | +0.00    |
| hills      | 0.50      | | desert     | +0.00    |
| wetlands   | 0.50      | | farmland   | +0.02    |
| mountains  | 0.60      | | woods      | +0.05    |
|            |           | | forest     | +0.10    |
|            |           | | jungle     | +0.10    |

### Water Types (Special Handling)

When crossing between land and water, a flat **0.20** multiplier is used regardless of the specific terrain types involved:
- `lakes` (always 0.20, even land↔land via lake)
- `coastal_ocean`
- `inland_sea`
- `narrows`

Water-to-water routes (e.g., sea tile to sea tile) use normal averaging (typically 0.40).

---

## Map Properties

| Property | Value |
|----------|-------|
| Dimensions | 16384 × 8192 pixels |
| Projection | Gall Stereographic |
| Equator Y | 3340 |

---

## Projection Distortion: The Hidden Penalty

The game calculates distances using raw pixel measurements on a **Gall Stereographic projection**. This projection preserves area reasonably well but **stretches vertical distances at high latitudes**. Since the game uses pixel distance rather than great-circle (spherical) distance, **locations far from the equator receive disproportionately large market access penalties**.

### Real-World Impact by Latitude

| Region | Latitude | km per Game Distance Unit |
|--------|----------|---------------------------|
| Somalia (Equator) | ~2°N | ~5.9 km |
| Zimbabwe | ~20°S | ~4.2 km |
| Constantinople | ~41°N | ~3.5 km |
| Finland | ~66°N | ~2.3 km |
| Northern Norway | ~70°N | ~1.6 km |

### Concrete Examples

#### Example 1: Similar Real Distances, Different Game Costs

| Route | Real Distance | Game Distance | Ratio |
|-------|---------------|---------------|-------|
| hudur → luuq (Somalia, 2°N) | 134 km | 22.58 | 5.9 km/unit |
| korvala → sodankyla (Finland, 66°N) | 108 km | 46.72 | 2.3 km/unit |

Despite being **20% shorter in real distance**, the Finland route costs **107% more** in game market access terms.

#### Example 2: Extreme Northern Penalty

| Route | Real Distance | Game Distance | Ratio |
|-------|---------------|---------------|-------|
| deatnu → vardo (Norway, 70°N) | 66 km | 40.65 | 1.6 km/unit |
| great_zimbabwe → naletale (Zimbabwe, 20°S) | 92 km | 22.06 | 4.2 km/unit |

The Norway route is **28% shorter** in reality but costs **84% more** in-game.

### Why This Happens

The Gall Stereographic projection maps the spherical Earth onto a flat rectangle. At the equator, one pixel represents approximately the same distance in all directions. But as you move toward the poles:

1. **Longitude lines converge** on a real globe (they meet at the poles)
2. **The map keeps them parallel**, spreading them apart artificially
3. **Pixel distances grow** relative to real-world distances

At 70°N latitude, the horizontal scale factor is roughly `cos(70°) ≈ 0.34`, meaning horizontal distances are **~3× larger in pixels** than they would be at the equator for the same real-world distance.

### Gameplay Implications

This projection distortion means:
- **Scandinavian trade networks** are mechanically disadvantaged
- **Russian expansion** faces steeper market access costs
- **Alaskan and Patagonian** regions are economically isolated beyond geographic reality
- **Equatorial regions** (Africa, Indonesia, Amazon) have relatively better market access

This is not necessarily a bug—it may be intentional game balance—but it's worth understanding when planning trade strategies.

---

## Validation

The formula was validated against **61 test cases** spanning:
- All latitudes from 55°S (Patagonia) to 71°N (Siberia)
- All terrain types (flatland, hills, wetlands, mountains, lakes, seas)
- Various movement directions (N-S, E-W, diagonal)
- Water↔land transitions

**Result: 61/61 matches (100%)** with error tolerance of ±1.0 game distance units.

---

## Appendix A: Project Files

### Required Files

| File | Purpose |
|------|---------|
| `test_data.txt` | Test case data (CSV format) |
| `generate_report.py` | Generates detailed test report |
| `map_data/location_templates.txt` | Game terrain data (28,573 locations) |

### Output Files

| File | Purpose |
|------|---------|
| `test_results.md` | Detailed per-test-case analysis |
| `EU5_MARKET_ACCESS_FORMULA.md` | This summary document |

### Analysis Scripts (Optional)

| File | Purpose |
|------|---------|
| `analyze_patterns.py` | Pattern analysis for debugging outliers |
| `analyze_norway.py` | Deep-dive on high-latitude behavior |

---

## Appendix B: Test Data Format

```
# Format: start_location, start_x, start_y, end_location, end_x, end_y, in_game_distance, notes
hudur, 9698, 3561, luuq, 9646, 3583, 22.58, Near equator
deatnu, 8968, 7801, vardo, 9049, 7794, 40.65, Far north E-W
```

Lines starting with `#` are comments. The `notes` field is optional.

---

## Appendix C: Coordinate Conversion

To convert pixel coordinates to latitude/longitude:

```python
import math

WIDTH = 16384
EQUATOR_Y = 3340
R_MAP = (WIDTH * math.sqrt(2)) / (2 * math.pi)
Y_FACTOR = R_MAP * (1 + math.sqrt(2) / 2)

def pixel_to_latlon(x, y):
    """Returns (latitude, longitude) in degrees"""
    x_proj = x - (WIDTH / 2)
    lon = math.degrees((x_proj * math.sqrt(2)) / R_MAP)
    y_proj = y - EQUATOR_Y
    lat = math.degrees(2 * math.atan(y_proj / Y_FACTOR))
    return lat, lon
```

---

## Appendix D: Distortion Factor Mathematics

### The Problem

The game uses pixel distance on a flat map, but the Earth is a sphere. At high latitudes, the same pixel distance represents a much smaller real-world distance because longitude lines converge toward the poles.

### Gall Stereographic Projection Equations

The projection maps spherical coordinates (latitude φ, longitude λ) to pixel coordinates (x, y):

```
x = R × (√2/2) × λ
y = R × (1 + √2/2) × tan(φ/2)
```

Where R is the map's scale constant and angles are in radians.

### Precise Distortion Calculation

**East-West (horizontal) movement:**

| Quantity | Formula |
|----------|---------|
| Pixel distance | Δx = R × (√2/2) × Δλ |
| Real distance | d = R_earth × cos(φ) × Δλ |
| Distortion vs equator | **D_EW = 1 / cos(φ)** |

**North-South (vertical) movement:**

| Quantity | Formula |
|----------|---------|
| Pixel distance | Δy ≈ R × (1 + √2/2) × Δφ / (2 × cos²(φ/2)) |
| Real distance | d = R_earth × Δφ |
| Distortion vs equator | **D_NS = 1 / cos²(φ/2)** |

**Combined (diagonal) movement:**

For movement with pixel components (Δx, Δy), calculate the corrected real-world distance:

```python
import math

def pixel_to_real_distance(dx, dy, lat_degrees):
    """
    Convert pixel displacement to approximate real-world distance ratio.
    Returns the distortion factor (pixel_dist / real_dist) relative to equator.
    """
    lat = math.radians(lat_degrees)

    # Corrected components
    real_dx = dx * math.cos(lat)           # E-W: shrinks by cos(lat)
    real_dy = dy * math.cos(lat/2)**2      # N-S: shrinks by cos²(lat/2)

    # Ratio of pixel distance to corrected distance
    pixel_dist = math.sqrt(dx**2 + dy**2)
    real_equiv = math.sqrt(real_dx**2 + real_dy**2)

    return pixel_dist / real_equiv if real_equiv > 0 else 1.0
```

### Handy Approximation

For quick estimates, **use the cosine of latitude**:

```
Distortion Factor ≈ 1 / cos(latitude)
```

This approximation works well for movement that has a significant E-W component, which is typical at high latitudes where the map is stretched horizontally.

| Latitude | cos(lat) | Distortion Factor | Interpretation |
|----------|----------|-------------------|----------------|
| 0° (equator) | 1.000 | 1.0× | Baseline |
| 20° | 0.940 | 1.1× | Minimal effect |
| 40° | 0.766 | 1.3× | Noticeable |
| 50° | 0.643 | 1.6× | Significant |
| 60° | 0.500 | 2.0× | Double penalty |
| 66° (Arctic Circle) | 0.407 | 2.5× | Severe |
| 70° | 0.342 | 2.9× | ~3× penalty |
| 80° | 0.174 | 5.8× | Extreme |

### Worked Example

**Route:** korvala → kittila (Finland, ~67°N)
- Pixel distance: √(73² + 55²) = 91.4 pixels
- Game distance: 91.4 × 0.60 (wetlands) = 54.84

**If the game used real-world distance:**
- Corrected dx: 73 × cos(67°) = 28.5
- Corrected dy: 55 × cos²(33.5°) = 38.3
- Corrected distance: √(28.5² + 38.3²) = 47.7
- Would-be game distance: 47.7 × 0.60 = 28.6

**The distortion penalty:** 54.84 / 28.6 = **1.92×** (nearly double!)

### Correcting for Latitude (Hypothetical Fix)

If the game wanted to use geographically-accurate distances, it could apply:

```python
def corrected_pixel_distance(x1, y1, x2, y2):
    """Calculate latitude-corrected pixel distance"""
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Get average latitude
    avg_y = (y1 + y2) / 2
    lat = pixel_to_lat(avg_y)  # Convert Y to latitude

    # Apply correction factors
    corrected_dx = dx * math.cos(math.radians(lat))
    corrected_dy = dy * math.cos(math.radians(lat/2))**2

    return math.sqrt(corrected_dx**2 + corrected_dy**2)
```

This would make Scandinavian and Russian trade networks competitive with equatorial regions.

---

*Analysis completed January 2026*
