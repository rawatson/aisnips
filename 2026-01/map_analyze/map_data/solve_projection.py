import math

# Data Points from reid_notes.md
# Format: (x1, y1), (x2, y2), in_game_dist
data = [
    ((9698, 3561), (9646, 3583), 22.58),  # Ajuraan (Near Eq)
    ((8698, 7801), (9049, 7794), 40.65),  # Norway (High Lat)
    ((9005, 5709), (8981, 5710), 10.08)   # Byzantine (Mid Lat)
]

WIDTH = 16384
HEIGHT = 8192
EQUATOR_Y = 3340

def solve_gall_stereographic(R_factor):
    # R_factor adjusts the theoretical R
    R = (WIDTH * math.sqrt(2)) / (2 * math.pi) * R_factor
    Y_FACTOR = R * (1 + math.sqrt(2)/2)
    
    print(f"\nTesting Gall Stereographic with R_factor={R_factor:.2f}")
    
    avg_ratio = 0
    ratios = []
    
    for (p1, p2, d_game) in data:
        # Convert to Lat/Lon
        def get_geo(x, y):
            x_proj = x - (WIDTH / 2)
            lon = (x_proj * math.sqrt(2)) / R
            
            y_proj = y - EQUATOR_Y
            lat = 2 * math.atan(y_proj / Y_FACTOR)
            return lat, lon # radians

        lat1, lon1 = get_geo(*p1)
        lat2, lon2 = get_geo(*p2)
        
        # Great Circle
        dLon = lon2 - lon1
        # haversine
        a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2) * math.sin(dLon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        # Distance on Unit Sphere
        dist_sphere = c
        
        ratio = d_game / dist_sphere
        ratios.append(ratio)
        print(f"  Game: {d_game:.2f} | Sphere: {dist_sphere:.4f} | Ratio: {ratio:.2f} | LatDeg: {math.degrees(lat1):.1f}")

    # Check consistency
    mean_ratio = sum(ratios) / len(ratios)
    variance = sum((r - mean_ratio)**2 for r in ratios)
    print(f"  Mean Ratio: {mean_ratio:.2f}, Variance: {variance:.4f}")

def solve_equirectangular():
    print(f"\nTesting Equirectangular")
    
    ratios = []
    
    for (p1, p2, d_game) in data:
        # Simple linear mapping
        # Lon = (x - W/2) / W * 2pi
        lon1 = (p1[0] - WIDTH/2) / WIDTH * 2 * math.pi
        lon2 = (p2[0] - WIDTH/2) / WIDTH * 2 * math.pi
        
        # Lat: Let's assume height covers 180 degrees? Or 140?
        # Let's try to infer Lat from Y
        # y_eq = 3340.
        # If pixels per degree is constant.
        # ppm = WIDTH / 360 = 45.51 px/deg
        ppm = WIDTH / (2*math.pi)
        
        lat1 = (p1[1] - EQUATOR_Y) / ppm
        lat2 = (p2[1] - EQUATOR_Y) / ppm
        
        dLon = lon2 - lon1
        a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2) * math.sin(dLon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        dist_sphere = c
        
        ratio = d_game / dist_sphere
        ratios.append(ratio)
        print(f"  Game: {d_game:.2f} | Sphere: {dist_sphere:.4f} | Ratio: {ratio:.2f} | LatDeg: {math.degrees(lat1):.1f}")
        
    mean_ratio = sum(ratios) / len(ratios)
    variance = sum((r - mean_ratio)**2 for r in ratios)
    print(f"  Mean Ratio: {mean_ratio:.2f}, Variance: {variance:.4f}")

solve_gall_stereographic(1.0)
solve_equirectangular()
