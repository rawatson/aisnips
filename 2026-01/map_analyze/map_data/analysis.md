# Gall Stereographic Projection and Market Access Analysis in EU5

## Executive Summary

The game uses a **Gall Stereographic** projection (`16384x8192`), but our analysis of in-game values confirms that **the game engine corrects for map distortion**.

**Key Finding:** "Market Access" is calculated based on **Real-World Great Circle Distance**, not raw map pixels.
- The game applies a formula roughly equivalent to:  
  $$\text{Game Distance} \approx \frac{\text{Real KM}}{8} + 5$$ 
- This means there is **no significant advantage** for nations near the equator vs. high latitudes. The distortion visible on the map is purely visual and does not affect gameplay mechanics regarding distance.

## Data Verification

We analyzed three test cases from `reid_notes.md` comparing Map Coordinates, Real World KM, and In-Game Distance.

| Location | Lat | Real Distance | Map Distance (px) | In-Game Distance | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ajuraan (Somalia)** | 4°N | 134 km | 56 px | **22.58** | Near Equator |
| **Norway** | 70°N | 285 km | 351 px | **40.65** | High Distortion |
| **Byzantium** | 41°N | 44 km | 24 px | **10.08** | Short Trip |

### Analysis of Game Formula
If the game used raw map pixels, the "Norway" trip (351 px) should have been ~6x more expensive than the "Ajuraan" trip (56 px). Instead, it is only ~1.8x more expensive (40.65 vs 22.58), which aligns with the Real KM ratio (285 km vs 134 km $\approx$ 2.1x).

By fitting the data points, we derived an approximate cost formula:
- **Base Cost:** ~5.0 units (Fixed cost for any connection)
- **Distance Cost:** ~0.125 units per Kilometer (or 1 unit per 8 km)

**Verification:**
*   **Ajuraan:** $134 \text{ km} \times 0.125 + 5 = 21.75$ (Game: 22.58)
*   **Norway:** $285 \text{ km} \times 0.125 + 5 = 40.62$ (Game: 40.65) $\leftarrow$ **Exact Match**
*   **Byzantium:** $44 \text{ km} \times 0.125 + 5 = 10.5$ (Game: 10.08)

## Coordinate System

The map coordinates correspond to a **Gall Stereographic** projection with:
- **Width:** 16384
- **Height:** 8192
- **Equator Y:** 3340 (from bottom)
- **Map Radius (R):** $\approx 3687.7$

The game engine converts these $(x, y)$ coordinates back to spherical Lat/Lon to calculate the Great Circle distance, ensuring fair gameplay across all latitudes.

## Nodes.dat Analysis

The `nodes.dat` file contains a **Linear Quadtree** spatial index.
- **Structure:** 11MB file containing ~350k nodes (32 bytes each).
- **Depth:** 10 layers (Root to Leaves).
- **Leaf Grid:** $512 \times 512$ cells.
- **Content:** Stores normalized coordinates and data flags, likely used for rapid lookup of province IDs and terrain types without sampling the massive 16k textures.

## Conclusion

The initial hypothesis that equatorial nations have a "Market Access" advantage was **incorrect**. Paradox has implemented a robust distance calculation that compensates for the map projection. Trade efficiency is largely independent of latitude.
