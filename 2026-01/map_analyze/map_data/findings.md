# Final Distance Cost Analysis

## The Formula
After analyzing 21 test cases, we have confirmed that the game calculates local movement cost using a **Weighted Average of Terrain Costs** applied to the **Raw Pixel Distance**.

$$ \text{Cost} = \text{Pixel Distance} \times \frac{\text{Cost}(\text{Start}) + \text{Cost}(\text{End})}{2} $$

This confirms that **Map Projection Distortion applies directly to movement costs.** Moving in the far North (where pixels are stretched) is significantly more expensive than moving at the Equator.

## Terrain Costs (Per Pixel)

### Base Terrain Costs
| Terrain Type | Cost Factor | Notes |
| :--- | :--- | :--- |
| **Flatland** | **0.40** | Standard Land movement. |
| **Sea / Coastal** | **0.40** | Standard Sea movement. |
| **Inland Sea** | **0.40** | Same as Ocean. |
| **Hills** | **0.50** | +25% cost vs Flatland. |
| **Wetlands** | **0.50** | Same as Hills. |
| **Plateau** | **0.45** | Slightly cheaper than Hills. |

### Vegetation Modifiers (Additive)
These values are added to the Base Terrain Cost.

| Vegetation | Modifier | Example |
| :--- | :--- | :--- |
| **Sparse / Grass** | **+0.00** | No penalty. |
| **Arid / Cold Arid**| **+0.00** | No penalty. |
| **Farmland** | **+0.02** | Negligible increase (or rounding error). |
| **Woods** | **+0.05** | Minor penalty. |
| **Forest** | **+0.10** | Significant penalty (+25% on Flatland). |

### Special Rules
*   **Lakes:** **0.20 Fixed.**
    *   Any movement *into* or *out of* a Lake (or between Lakes) uses a fixed factor of **0.20**, overriding terrain costs. This makes Lakes effectively "Highways".
    
## Anomalies
*   **Test 2 (Deatnu -> Vardo):** This remains the only outlier, costing ~4x less than predicted by pixels. It matches the **Great Circle** formula exactly. This suggests that for **Non-Adjacent** or **Long-Distance** pathfinding (or specific strait crossings), the game falls back to a spherical calculation to avoid massive penalties.

## Summary Table of Verified Costs
| Case | Start | End | Predicted | Actual | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | Flatland (0.4) | Flatland (0.4) | **0.40** | 0.40 | Match |
| **Rough** | Hills (0.5) | Hills (0.5) | **0.50** | 0.50 | Match |
| **Mixed** | Flatland (0.4) | Hills (0.5) | **0.45** | 0.45 | Match |
| **Forest** | Flat (0.4)+For(0.1) | Flat (0.4)+For(0.1) | **0.50** | 0.50 | Match |
| **Lake** | Flatland | Lake | **0.20** | 0.20 | Match |