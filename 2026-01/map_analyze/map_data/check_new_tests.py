import math

# Constants
WIDTH = 16384
HEIGHT = 8192
EQUATOR_Y = 3340

# Terrain Costs
BASE_COSTS = {
    "flatland": 0.40,
    "coastal_ocean": 0.40,
    "inland_sea": 0.40,
    "hills": 0.50,
    "wetlands": 0.50,
    "plateau": 0.45,
    "lakes": 0.20 # Special
}

VEG_COSTS = {
    "forest": 0.10,
    "woods": 0.05,
    "farmland": 0.02,
    "sparse": 0.00,
    "grasslands": 0.00,
    "arid": 0.00,
    "cold_arid": 0.00
}

# Location Data (Manually enriched)
loc_data = {
    "hudur": {"topo": "flatland", "veg": "grasslands"},
    "luuq": {"topo": "flatland", "veg": "grasslands"},
    "deatnu": {"topo": "hills", "veg": "sparse"},
    "vardo": {"topo": "hills", "veg": "sparse"},
    "constantinople": {"topo": "flatland", "veg": "farmland"},
    "silivri": {"topo": "flatland", "veg": "farmland"},
    "kola": {"topo": "wetlands", "veg": "sparse"},
    "maaselka": {"topo": "hills", "veg": "sparse"},
    "loimola": {"topo": "flatland", "veg": "forest"},
    "salmi": {"topo": "flatland", "veg": "forest"},
    "lake_ladoga": {"topo": "lakes", "veg": "sparse"},
    "intsi_cape": {"topo": "inland_sea", "veg": "sparse"},
    "varzuga_estuary": {"topo": "inland_sea", "veg": "sparse"},
    "inner_onega_bay": {"topo": "inland_sea", "veg": "sparse"},
    "kirkenes": {"topo": "hills", "veg": "sparse"},
    "varanger": {"topo": "hills", "veg": "sparse"},
    "gonen": {"topo": "flatland", "veg": "woods"},
    "lake_manyas": {"topo": "lakes", "veg": "sparse"},
    "bandirma": {"topo": "hills", "veg": "forest"},
    "pyalitsa": {"topo": "wetlands", "veg": "sparse"},
    "ponoi": {"topo": "wetlands", "veg": "sparse"},
    "cadde_cape": {"topo": "coastal_ocean", "veg": "sparse"},
    "banaadir_coast": {"topo": "coastal_ocean", "veg": "sparse"},
    "great_zimbabwe": {"topo": "plateau", "veg": "forest"},
    "mabveni": {"topo": "plateau", "veg": "forest"},
    "juni_aiken": {"topo": "flatland", "veg": "grasslands"},
    "ciaike": {"topo": "flatland", "veg": "grasslands"},
    "nakenk": {"topo": "flatland", "veg": "grasslands"},
    "hamenk": {"topo": "wetlands", "veg": "grasslands"},
    "kauwes": {"topo": "flatland", "veg": "grasslands"},
    "utumaala": {"topo": "hills", "veg": "forest"},
    "wakimaala": {"topo": "hills", "veg": "forest"},
    "aguirre_bay": {"topo": "coastal_ocean", "veg": "sparse"},
    "blossom_bay": {"topo": "coastal_ocean", "veg": "sparse"},
    "corpen_aike": {"topo": "flatland", "veg": "sparse"},
    "chalten": {"topo": "hills", "veg": "sparse"},
}

cases = [
    (1, "hudur", "luuq", (9698, 3561), (9646, 3583), 22.58),
    (2, "deatnu", "vardo", (8698, 7801), (9049, 7794), 40.65),
    (3, "constantinople", "silivri", (9005, 5709), (8981, 5710), 10.08),
    (4, "kola", "maaselka", (9189, 7666), (9209, 7598), 35.44),
    (5, "loimola", "salmi", (9125, 7118), (9141, 7101), 11.67),
    (6, "loimola", "lake_ladoga", (9125, 7118), (9122, 7043), 15.01),
    (7, "intsi_cape", "varzuga_estuary", (9461, 7394), (9273, 7447), 78.13),
    (8, "inner_onega_bay", "varzuga_estuary", (9352, 7319), (9273, 7447), 60.16),
    (9, "kirkenes", "varanger", (9041, 7715), (9010, 7759), 26.91),
    (10, "gonen", "lake_manyas", (8952, 5647), (8964, 5650), 2.47),
    (11, "lake_manyas", "bandirma", (8964, 5650), (8967, 5662), 2.47),
    (12, "gonen", "bandirma", (8952, 5647), (8967, 5662), 11.13),
    (13, "pyalitsa", "ponoi", (9521, 7479), (9520, 7537), 29.00),
    (14, "cadde_cape", "banaadir_coast", (9808, 3472), (9762, 3433), 24.12),
    (15, "great_zimbabwe", "mabveni", (9095, 2239), (9082, 2252), 10.11),
    (16, "juni_aiken", "ciaike", (4478, 296), (4516, 264), 19.87),
    (17, "nakenk", "hamenk", (4619, 126), (4587, 136), 15.08),
    (18, "aguirre_bay", "blossom_bay", (4693, 42), (4794, 69), 41.81),
    (19, "utumaala", "wakimaala", (4627, 73), (4593, 76), 20.47),
    (20, "hamenk", "kauwes", (4587, 136), (4554, 142), 15.09),
    (21, "corpen_aike", "chalten", (4526, 441), (4437, 439), 40.06),
]

print(f"{'ID':<3} | {'Type':<15} | {'PxDist':<8} | {'Actual':<6} | {'Pred':<6} | {'Diff':<6} | {'Notes'}")
print("-" * 80)

for (id, s_name, e_name, p1, p2, actual) in cases:
    # Get properties
    s = loc_data.get(s_name, {"topo":"flatland","veg":"sparse"})
    e = loc_data.get(e_name, {"topo":"flatland","veg":"sparse"})
    
    # Calculate Cost Factors
    cost_s = BASE_COSTS.get(s['topo'], 0.4) + VEG_COSTS.get(s['veg'], 0.0)
    cost_e = BASE_COSTS.get(e['topo'], 0.4) + VEG_COSTS.get(e['veg'], 0.0)
    
    # Average
    avg_cost = (cost_s + cost_e) / 2.0
    
    # Lake Override
    if s['topo'] == 'lakes' or e['topo'] == 'lakes':
        avg_cost = 0.20
        
    # Pixel Dist
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    px_dist = math.sqrt(dx*dx + dy*dy)
    
    pred = px_dist * avg_cost
    
    diff = actual - pred
    
    note = "OK" if abs(diff) < 0.5 else "FAIL"
    if id == 2: note = "GC/Outlier"
    
    print(f"{id:<3} | {s['topo'][:3]}-{e['topo'][:3]} | {px_dist:<8.2f} | {actual:<6.2f} | {pred:<6.2f} | {diff:<6.2f} | {note}")
