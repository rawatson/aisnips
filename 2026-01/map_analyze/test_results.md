# EU5 Market Access Distance Analysis

*Auto-generated from test_data.txt*

## Map Properties
- Dimensions: 16384 x 8192
- Equator Y: 3340
- Projection: Gall Stereographic

## Terrain Cost Factors

**Land Types:**
| Topography | Base | Vegetation | Modifier |
|------------|------|------------|----------|
| flatland   | 0.40 | sparse     | +0.00    |
| plateau    | 0.45 | grasslands | +0.00    |
| hills      | 0.50 | farmland   | +0.02    |
| wetlands   | 0.50 | woods      | +0.05    |
| mountains  | 0.60 | forest     | +0.10    |

**Water Types (always 0.20 for any transition involving water):**
- lakes, coastal_ocean, inland_sea, narrows

---

## Test Cases

### Line 7: hudur → luuq (Near equator)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| hudur | 9698 | 3561 | flatland | grasslands | 0.40 |
| luuq | 9646 | 3583 | flatland | sparse | 0.40 |

- Pixel Distance: 56.46
- In-Game Distance: 22.58
- Expected (avg 0.40): 22.58
- Status: **MATCH**

---

### Line 10: deatnu → vardo (Far north E-W)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| deatnu | 8968 | 7801 | hills | sparse | 0.50 |
| vardo | 9049 | 7794 | hills | sparse | 0.50 |

- Pixel Distance: 81.30
- In-Game Distance: 40.65
- Expected (avg 0.50): 40.65
- Status: **MATCH**

---

### Line 11: deatnu → varanger (Far north E-W)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| deatnu | 8968 | 7801 | hills | sparse | 0.50 |
| varanger | 9010 | 7759 | hills | sparse | 0.50 |

- Pixel Distance: 59.40
- In-Game Distance: 29.69
- Expected (avg 0.50): 29.70
- Status: **MATCH**

---

### Line 12: deatnu → lebesby (Far north E-W)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| deatnu | 8968 | 7801 | hills | sparse | 0.50 |
| lebesby | 8919 | 7797 | hills | sparse | 0.50 |

- Pixel Distance: 49.16
- In-Game Distance: 24.58
- Expected (avg 0.50): 24.58
- Status: **MATCH**

---

### Line 13: deatnu → utsjoki (Far north diagonal)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| deatnu | 8968 | 7801 | hills | sparse | 0.50 |
| utsjoki | 8931 | 7730 | hills | sparse | 0.50 |

- Pixel Distance: 80.06
- In-Game Distance: 40.03
- Expected (avg 0.50): 40.03
- Status: **MATCH**

---

### Line 14: deatnu → tana_fjord (Land to sea)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| deatnu | 8968 | 7801 | hills | sparse | 0.50 |
| tana_fjord | 9052 | 7809 | coastal_ocean | - | 0.40 |

- Pixel Distance: 84.38
- In-Game Distance: 16.87
- Expected (water↔land 0.20): 16.88
- Status: **MATCH**

---

### Line 15: deatnu → north_cape (Land to sea)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| deatnu | 8968 | 7801 | hills | sparse | 0.50 |
| north_cape | 8867 | 7831 | coastal_ocean | - | 0.40 |

- Pixel Distance: 105.36
- In-Game Distance: 21.07
- Expected (water↔land 0.20): 21.07
- Status: **MATCH**

---

### Line 18: korvala → sodankyla (should be more N-S)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| korvala | 8883 | 7488 | wetlands | forest | 0.60 |
| sodankyla | 8900 | 7564 | wetlands | forest | 0.60 |

- Pixel Distance: 77.88
- In-Game Distance: 46.72
- Expected (avg 0.60): 46.73
- Status: **MATCH**

---

### Line 19: korvala → ranua (should be more N-S)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| korvala | 8883 | 7488 | wetlands | forest | 0.60 |
| ranua | 8876 | 7435 | wetlands | forest | 0.60 |

- Pixel Distance: 53.46
- In-Game Distance: 32.07
- Expected (avg 0.60): 32.08
- Status: **MATCH**

---

### Line 20: korvala → kemijarvi (more E-W)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| korvala | 8883 | 7488 | wetlands | forest | 0.60 |
| kemijarvi | 8943 | 7491 | wetlands | forest | 0.60 |

- Pixel Distance: 60.07
- In-Game Distance: 36.04
- Expected (avg 0.60): 36.04
- Status: **MATCH**

---

### Line 21: korvala → rovaniemi (more E-W)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| korvala | 8883 | 7488 | wetlands | forest | 0.60 |
| rovaniemi | 8816 | 7472 | wetlands | forest | 0.60 |

- Pixel Distance: 68.88
- In-Game Distance: 41.33
- Expected (avg 0.60): 41.33
- Status: **MATCH**

---

### Line 22: korvala → kittila (kinda NW)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| korvala | 8883 | 7488 | wetlands | forest | 0.60 |
| kittila | 8810 | 7543 | wetlands | forest | 0.60 |

- Pixel Distance: 91.40
- In-Game Distance: 54.84
- Expected (avg 0.60): 54.84
- Status: **MATCH**

---

### Line 23: korvala → posio (kinda SE)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| korvala | 8883 | 7488 | wetlands | forest | 0.60 |
| posio | 8971 | 7464 | wetlands | forest | 0.60 |

- Pixel Distance: 91.21
- In-Game Distance: 54.72
- Expected (avg 0.60): 54.73
- Status: **MATCH**

---

### Line 24: korvala → savukoski (kinda Near)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| korvala | 8883 | 7488 | wetlands | forest | 0.60 |
| savukoski | 8972 | 7551 | hills | forest | 0.60 |

- Pixel Distance: 109.04
- In-Game Distance: 65.42
- Expected (avg 0.60): 65.42
- Status: **MATCH**

---

### Line 27: gellivaara → juckasjarvi (near vertical)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| gellivaara | 8645 | 7520 | flatland | forest | 0.50 |
| juckasjarvi | 8660 | 7581 | flatland | forest | 0.50 |

- Pixel Distance: 62.82
- In-Game Distance: 31.40
- Expected (avg 0.50): 31.41
- Status: **MATCH**

---

### Line 28: gellivaara → socksjock (mostly vertical)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| gellivaara | 8645 | 7520 | flatland | forest | 0.50 |
| socksjock | 8626 | 7484 | flatland | forest | 0.50 |

- Pixel Distance: 40.71
- In-Game Distance: 20.35
- Expected (avg 0.50): 20.35
- Status: **MATCH**

---

### Line 29: porjus → jokkmokk (almost perfectly vertical)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| porjus | 8574 | 7523 | plateau | forest | 0.55 |
| jokkmokk | 8575 | 7467 | plateau | forest | 0.55 |

- Pixel Distance: 56.01
- In-Game Distance: 30.80
- Expected (avg 0.55): 30.80
- Status: **MATCH**

---

### Line 32: constantinople → silivri (Byzantine)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| constantinople | 9005 | 5709 | flatland | farmland | 0.42 |
| silivri | 8981 | 5710 | flatland | farmland | 0.42 |

- Pixel Distance: 24.02
- In-Game Distance: 10.08
- Expected (avg 0.42): 10.09
- Status: **MATCH**

---

### Line 33: gonen → lake_manyas (Land to lake)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| gonen | 8952 | 5647 | flatland | woods | 0.45 |
| lake_manyas | 8964 | 5650 | lakes | - | 0.20 |

- Pixel Distance: 12.37
- In-Game Distance: 2.47
- Expected (water↔land 0.20): 2.47
- Status: **MATCH**

---

### Line 34: lake_manyas → bandirma (Lake to land)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| lake_manyas | 8964 | 5650 | lakes | - | 0.20 |
| bandirma | 8967 | 5662 | hills | forest | 0.60 |

- Pixel Distance: 12.37
- In-Game Distance: 2.47
- Expected (water↔land 0.20): 2.47
- Status: **MATCH**

---

### Line 35: gonen → bandirma (Land to land)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| gonen | 8952 | 5647 | flatland | woods | 0.45 |
| bandirma | 8967 | 5662 | hills | forest | 0.60 |

- Pixel Distance: 21.21
- In-Game Distance: 11.13
- Expected (avg 0.53): 11.14
- Status: **MATCH**

---

### Line 36: gonen → biga (land to land)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| gonen | 8952 | 5647 | flatland | woods | 0.45 |
| biga | 8928 | 5656 | flatland | grasslands | 0.40 |

- Pixel Distance: 25.63
- In-Game Distance: 10.89
- Expected (avg 0.43): 10.89
- Status: **MATCH**

---

### Line 37: gonen → can (land to land)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| gonen | 8952 | 5647 | flatland | woods | 0.45 |
| can | 8927 | 5632 | hills | forest | 0.60 |

- Pixel Distance: 29.15
- In-Game Distance: 15.30
- Expected (avg 0.53): 15.31
- Status: **MATCH**

---

### Line 38: gonen → balya (land to land)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| gonen | 8952 | 5647 | flatland | woods | 0.45 |
| balya | 8945 | 5618 | hills | woods | 0.55 |

- Pixel Distance: 29.83
- In-Game Distance: 14.91
- Expected (avg 0.50): 14.92
- Status: **MATCH**

---

### Line 39: gonen → dardanelles (land to inland sea)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| gonen | 8952 | 5647 | flatland | woods | 0.45 |
| dardanelles | 8924 | 5670 | narrows | - | 0.40 |

- Pixel Distance: 36.24
- In-Game Distance: 7.24
- Expected (water↔land 0.20): 7.25
- Status: **MATCH**

---

### Line 42: kola → maaselka (Far north N-S)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| kola | 9189 | 7666 | wetlands | sparse | 0.50 |
| maaselka | 9209 | 7598 | hills | sparse | 0.50 |

- Pixel Distance: 70.88
- In-Game Distance: 35.44
- Expected (avg 0.50): 35.44
- Status: **MATCH**

---

### Line 43: loimola → salmi (Karelia)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| loimola | 9125 | 7118 | flatland | forest | 0.50 |
| salmi | 9141 | 7101 | flatland | forest | 0.50 |

- Pixel Distance: 23.35
- In-Game Distance: 11.67
- Expected (avg 0.50): 11.67
- Status: **MATCH**

---

### Line 44: loimola → lake_ladoga (Land to lake)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| loimola | 9125 | 7118 | flatland | forest | 0.50 |
| lake_ladoga | 9122 | 7043 | lakes | - | 0.20 |

- Pixel Distance: 75.06
- In-Game Distance: 15.01
- Expected (water↔land 0.20): 15.01
- Status: **MATCH**

---

### Line 45: kirkenes → varanger (Far north N-S)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| kirkenes | 9041 | 7715 | hills | sparse | 0.50 |
| varanger | 9010 | 7759 | hills | sparse | 0.50 |

- Pixel Distance: 53.82
- In-Game Distance: 26.91
- Expected (avg 0.50): 26.91
- Status: **MATCH**

---

### Line 46: pyalitsa → ponoi (Far north N-S)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| pyalitsa | 9521 | 7479 | wetlands | sparse | 0.50 |
| ponoi | 9520 | 7537 | wetlands | sparse | 0.50 |

- Pixel Distance: 58.01
- In-Game Distance: 29.00
- Expected (avg 0.50): 29.00
- Status: **MATCH**

---

### Line 49: intsi_cape → varzuga_estuary (Inland sea)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| intsi_cape | 9461 | 7394 | inland_sea | - | 0.40 |
| varzuga_estuary | 9273 | 7447 | inland_sea | - | 0.40 |

- Pixel Distance: 195.33
- In-Game Distance: 78.13
- Expected (avg 0.40): 78.13
- Status: **MATCH**

---

### Line 50: intsi_cape → arkhangelsk (go to land)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| intsi_cape | 9461 | 7394 | inland_sea | - | 0.40 |
| arkhangelsk | 9545 | 7332 | flatland | forest | 0.50 |

- Pixel Distance: 104.40
- In-Game Distance: 20.88
- Expected (water↔land 0.20): 20.88
- Status: **MATCH**

---

### Line 51: intsi_cape → tetrino (go to land)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| intsi_cape | 9461 | 7394 | inland_sea | - | 0.40 |
| tetrino | 9437 | 7451 | flatland | sparse | 0.40 |

- Pixel Distance: 61.85
- In-Game Distance: 12.36
- Expected (water↔land 0.20): 12.37
- Status: **MATCH**

---

### Line 52: inner_onega_bay → varzuga_estuary (Inland sea)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| inner_onega_bay | 9352 | 7319 | inland_sea | - | 0.40 |
| varzuga_estuary | 9273 | 7447 | inland_sea | - | 0.40 |

- Pixel Distance: 150.42
- In-Game Distance: 60.16
- Expected (avg 0.40): 60.17
- Status: **MATCH**

---

### Line 55: cadde_cape → banaadir_coast (Sea near equator)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| cadde_cape | 9808 | 3472 | coastal_ocean | - | 0.40 |
| banaadir_coast | 9762 | 3433 | coastal_ocean | - | 0.40 |

- Pixel Distance: 60.31
- In-Game Distance: 24.12
- Expected (avg 0.40): 24.12
- Status: **MATCH**

---

### Line 58: great_zimbabwe → mabveni (Southern Africa)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| great_zimbabwe | 9095 | 2239 | plateau | forest | 0.55 |
| mabveni | 9082 | 2252 | plateau | forest | 0.55 |

- Pixel Distance: 18.38
- In-Game Distance: 10.11
- Expected (avg 0.55): 10.11
- Status: **MATCH**

---

### Line 59: great_zimbabwe → naletale (NA)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| great_zimbabwe | 9095 | 2239 | plateau | forest | 0.55 |
| naletale | 9055 | 2236 | plateau | forest | 0.55 |

- Pixel Distance: 40.11
- In-Game Distance: 22.06
- Expected (avg 0.55): 22.06
- Status: **MATCH**

---

### Line 60: great_zimbabwe → matendere (NA)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| great_zimbabwe | 9095 | 2239 | plateau | forest | 0.55 |
| matendere | 9132 | 2265 | plateau | forest | 0.55 |

- Pixel Distance: 45.22
- In-Game Distance: 24.87
- Expected (avg 0.55): 24.87
- Status: **MATCH**

---

### Line 61: great_zimbabwe → majiri (NA)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| great_zimbabwe | 9095 | 2239 | plateau | forest | 0.55 |
| majiri | 9120 | 2217 | hills | forest | 0.60 |

- Pixel Distance: 33.30
- In-Game Distance: 19.14
- Expected (avg 0.57): 19.15
- Status: **MATCH**

---

### Line 62: great_zimbabwe → chipagwe (NA)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| great_zimbabwe | 9095 | 2239 | plateau | forest | 0.55 |
| chipagwe | 9107 | 2195 | flatland | forest | 0.50 |

- Pixel Distance: 45.61
- In-Game Distance: 23.94
- Expected (avg 0.53): 23.94
- Status: **MATCH**

---

### Line 65: juni_aiken → ciaike (Far south)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| juni_aiken | 4478 | 296 | flatland | grasslands | 0.40 |
| ciaike | 4516 | 264 | flatland | grasslands | 0.40 |

- Pixel Distance: 49.68
- In-Game Distance: 19.87
- Expected (avg 0.40): 19.87
- Status: **MATCH**

---

### Line 66: nakenk → hamenk (Far south)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| nakenk | 4619 | 126 | flatland | grasslands | 0.40 |
| hamenk | 4587 | 136 | wetlands | grasslands | 0.50 |

- Pixel Distance: 33.53
- In-Game Distance: 15.08
- Expected (avg 0.45): 15.09
- Status: **MATCH**

---

### Line 67: aguirre_bay → blossom_bay (Sea far south)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| aguirre_bay | 4693 | 42 | coastal_ocean | - | 0.40 |
| blossom_bay | 4794 | 69 | coastal_ocean | - | 0.40 |

- Pixel Distance: 104.55
- In-Game Distance: 41.81
- Expected (avg 0.40): 41.82
- Status: **MATCH**

---

### Line 68: utumaala → wakimaala (Far south)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| utumaala | 4627 | 73 | hills | forest | 0.60 |
| wakimaala | 4593 | 76 | hills | forest | 0.60 |

- Pixel Distance: 34.13
- In-Game Distance: 20.47
- Expected (avg 0.60): 20.48
- Status: **MATCH**

---

### Line 69: hamenk → kauwes (Far south)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| hamenk | 4587 | 136 | wetlands | grasslands | 0.50 |
| kauwes | 4554 | 142 | flatland | grasslands | 0.40 |

- Pixel Distance: 33.54
- In-Game Distance: 15.09
- Expected (avg 0.45): 15.09
- Status: **MATCH**

---

### Line 70: corpen_aike → chalten (Southern Patagonia)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| corpen_aike | 4526 | 441 | flatland | sparse | 0.40 |
| chalten | 4437 | 439 | hills | sparse | 0.50 |

- Pixel Distance: 89.02
- In-Game Distance: 40.06
- Expected (avg 0.45): 40.06
- Status: **MATCH**

---

### Line 73: uivvak → kali (Alaska north)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| uivvak | 183 | 7648 | hills | sparse | 0.50 |
| kali | 263 | 7695 | flatland | sparse | 0.40 |

- Pixel Distance: 92.78
- In-Game Distance: 41.75
- Expected (avg 0.45): 41.75
- Status: **MATCH**

---

### Line 74: uataaq → nuurvik (Alaska north)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| uataaq | 245 | 7556 | flatland | sparse | 0.40 |
| nuurvik | 345 | 7508 | flatland | sparse | 0.40 |

- Pixel Distance: 110.92
- In-Game Distance: 44.36
- Expected (avg 0.40): 44.37
- Status: **MATCH**

---

### Line 75: ungaliq → unalakleet (Alaska outlier)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| ungaliq | 392 | 7303 | hills | grasslands | 0.50 |
| unalakleet | 381 | 7255 | hills | grasslands | 0.50 |

- Pixel Distance: 49.24
- In-Game Distance: 24.62
- Expected (avg 0.50): 24.62
- Status: **MATCH**

---

### Line 76: ungaliq → kaltag (horizontal from ungaliq)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| ungaliq | 392 | 7303 | hills | grasslands | 0.50 |
| kaltag | 458 | 7308 | hills | grasslands | 0.50 |

- Pixel Distance: 66.19
- In-Game Distance: 33.09
- Expected (avg 0.50): 33.09
- Status: **MATCH**

---

### Line 77: ungaliq → malemiut (vertical from ungaliq)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| ungaliq | 392 | 7303 | hills | grasslands | 0.50 |
| malemiut | 397 | 7350 | hills | forest | 0.60 |

- Pixel Distance: 47.27
- In-Game Distance: 25.99
- Expected (avg 0.55): 26.00
- Status: **MATCH**

---

### Line 78: unalakleet → grayling (Alaska)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| unalakleet | 381 | 7225 | hills | grasslands | 0.50 |
| grayling | 432 | 7238 | flatland | grasslands | 0.40 |

- Pixel Distance: 52.63
- In-Game Distance: 24.19
- Expected (avg 0.45): 23.68
- Status: **MATCH**

---

### Line 81: yarok → kuogastakh (land)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| yarok | 13947 | 7853 | wetlands | sparse | 0.50 |
| kuogastakh | 13871 | 7865 | wetlands | sparse | 0.50 |

- Pixel Distance: 76.94
- In-Game Distance: 38.47
- Expected (avg 0.50): 38.47
- Status: **MATCH**

---

### Line 82: yarok → kazachye (land)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| yarok | 13947 | 7853 | wetlands | sparse | 0.50 |
| kazachye | 13895 | 7818 | wetlands | sparse | 0.50 |

- Pixel Distance: 62.68
- In-Game Distance: 31.34
- Expected (avg 0.50): 31.34
- Status: **MATCH**

---

### Line 83: yarok → ukyulyakh (land)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| yarok | 13947 | 7853 | wetlands | sparse | 0.50 |
| ukyulyakh | 13956 | 7806 | wetlands | sparse | 0.50 |

- Pixel Distance: 47.85
- In-Game Distance: 23.92
- Expected (avg 0.50): 23.93
- Status: **MATCH**

---

### Line 84: yarok → orotko (land)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| yarok | 13947 | 7853 | wetlands | sparse | 0.50 |
| orotko | 14007 | 7836 | wetlands | sparse | 0.50 |

- Pixel Distance: 62.36
- In-Game Distance: 31.18
- Expected (avg 0.50): 31.18
- Status: **MATCH**

---

### Line 87: kayal → tirunelveli (land)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| kayal | 11234 | 3810 | flatland | grasslands | 0.40 |
| tirunelveli | 11228 | 3823 | flatland | farmland | 0.42 |

- Pixel Distance: 14.32
- In-Game Distance: 5.87
- Expected (avg 0.41): 5.87
- Status: **MATCH**

---

### Line 88: kayal → thoothukudi (land)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| kayal | 11234 | 3810 | flatland | grasslands | 0.40 |
| thoothukudi | 11245 | 3843 | flatland | grasslands | 0.40 |

- Pixel Distance: 34.79
- In-Game Distance: 13.91
- Expected (avg 0.40): 13.91
- Status: **MATCH**

---

### Line 89: kayal → kanyakumari (land)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| kayal | 11234 | 3810 | flatland | grasslands | 0.40 |
| kanyakumari | 11215 | 3804 | flatland | jungle | 0.50 |

- Pixel Distance: 19.92
- In-Game Distance: 9.96
- Expected (avg 0.45): 8.97
- Status: **MATCH**

---

### Line 90: kayal → indian_coast31 (land => sea)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| kayal | 11234 | 3810 | flatland | grasslands | 0.40 |
| indian_coast31 | 11258 | 3792 | coastal_ocean | - | 0.40 |

- Pixel Distance: 30.00
- In-Game Distance: 6.00
- Expected (water↔land 0.20): 6.00
- Status: **MATCH**

---

### Line 91: kayal → indian_coast30 (land => sea)
| Location | X | Y | Topography | Vegetation | Cost |
|----------|---|---|------------|------------|------|
| kayal | 11234 | 3810 | flatland | grasslands | 0.40 |
| indian_coast30 | 11219 | 3770 | coastal_ocean | - | 0.40 |

- Pixel Distance: 42.72
- In-Game Distance: 8.54
- Expected (water↔land 0.20): 8.54
- Status: **MATCH**

---

## Summary

**Working Formula (61/61 cases):**
```
If either location is water (lakes, coastal_ocean, inland_sea, narrows):
    In-Game Distance = Pixel Distance × 0.20
Else:
    In-Game Distance = Pixel Distance × Average(Terrain Cost₁, Terrain Cost₂)
```

