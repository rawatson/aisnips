import math

WIDTH = 16384
HEIGHT = 8192
EQUATOR_Y = 3340
EARTH_RADIUS_KM = 6371.0

R_map = (WIDTH * math.sqrt(2)) / (2 * math.pi)
Y_FACTOR = R_map * (1 + math.sqrt(2)/2)

def pixel_to_latlon(x, y):
    x_proj = x - (WIDTH / 2)
    lon = (x_proj * math.sqrt(2)) / R_map
    y_proj = y - EQUATOR_Y
    lat = 2 * math.atan(y_proj / Y_FACTOR)
    return lat, lon

def great_circle_km(lat1, lon1, lat2, lon2):
    dLon = lon2 - lon1
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2) * math.sin(dLon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return EARTH_RADIUS_KM * c

def pixel_distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx*dx + dy*dy)

# All Norway outlier tests
norway_tests = [
    (2, "deatnu", "vardo", (8698, 7801), (9049, 7794), 40.65),
    (26, "deatnu", "lebesby", (8698, 7801), (8919, 7797), 24.58),
    (27, "deatnu", "utsjoki", (8698, 7801), (8931, 7730), 40.03),
    (28, "deatnu", "tana_fjord", (8698, 7801), (9052, 7809), 16.87),
    (29, "deatnu", "north_cape", (8698, 7801), (8867, 7831), 21.07),
]

print("=" * 110)
print("NORWAY OUTLIER ANALYSIS")
print("=" * 110)

print(f"\n{'ID':<3} | {'Route':<20} | {'PxDist':<8} | {'Actual':<7} | {'GC km':<7} | {'GC/7':<7} | {'GC/8+5':<7} | {'dx':<5} | {'dy':<5} | {'Lat°':<6}")
print("-" * 110)

for id, start, end, p1, p2, actual in norway_tests:
    px_dist = pixel_distance(p1, p2)
    lat1, lon1 = pixel_to_latlon(p1[0], p1[1])
    lat2, lon2 = pixel_to_latlon(p2[0], p2[1])
    gc_km = great_circle_km(lat1, lon1, lat2, lon2)

    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    avg_lat = math.degrees((lat1 + lat2) / 2)

    gc_7 = gc_km / 7
    gc_8_5 = gc_km / 8 + 5

    print(f"{id:<3} | {start}->{end:<10} | {px_dist:<8.2f} | {actual:<7.2f} | {gc_km:<7.1f} | {gc_7:<7.2f} | {gc_8_5:<7.2f} | {dx:<5} | {dy:<5} | {avg_lat:<6.1f}")

print("\n" + "=" * 110)
print("HYPOTHESIS TESTING")
print("=" * 110)

print("\n--- Testing: actual = gc_km * k ---")
for id, start, end, p1, p2, actual in norway_tests:
    lat1, lon1 = pixel_to_latlon(p1[0], p1[1])
    lat2, lon2 = pixel_to_latlon(p2[0], p2[1])
    gc_km = great_circle_km(lat1, lon1, lat2, lon2)
    k = actual / gc_km
    print(f"  Test {id}: k = {k:.4f}")

print("\n--- Testing: actual = gc_km / d + offset ---")
# Try to find best d and offset
for offset in [0, 2, 5, 8]:
    print(f"\n  With offset = {offset}:")
    for id, start, end, p1, p2, actual in norway_tests:
        lat1, lon1 = pixel_to_latlon(p1[0], p1[1])
        lat2, lon2 = pixel_to_latlon(p2[0], p2[1])
        gc_km = great_circle_km(lat1, lon1, lat2, lon2)
        if actual > offset:
            d = gc_km / (actual - offset)
            print(f"    Test {id}: d = {d:.2f}")

print("\n--- Testing: actual = px_dist * cos(lat) * terrain ---")
for id, start, end, p1, p2, actual in norway_tests:
    px_dist = pixel_distance(p1, p2)
    lat1, lon1 = pixel_to_latlon(p1[0], p1[1])
    lat2, lon2 = pixel_to_latlon(p2[0], p2[1])
    avg_lat = (lat1 + lat2) / 2
    cos_lat = math.cos(avg_lat)

    # What terrain factor would be needed?
    implied_terrain = actual / (px_dist * cos_lat)
    print(f"  Test {id}: cos(lat)={cos_lat:.4f}, implied terrain={implied_terrain:.4f}")

print("\n--- Testing: Corrected horizontal distance ---")
for id, start, end, p1, p2, actual in norway_tests:
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])

    lat1, lon1 = pixel_to_latlon(p1[0], p1[1])
    lat2, lon2 = pixel_to_latlon(p2[0], p2[1])
    avg_lat = (lat1 + lat2) / 2
    cos_lat = math.cos(avg_lat)

    corrected_dx = dx * cos_lat
    corrected_dist = math.sqrt(corrected_dx**2 + dy**2)

    # What terrain factor would give actual?
    implied_terrain = actual / corrected_dist
    print(f"  Test {id}: corrected_dist={corrected_dist:.2f}, implied terrain={implied_terrain:.4f}")

print("\n" + "=" * 110)
print("COMPARISON: Norway vs Working High-Latitude Tests")
print("=" * 110)

working_high_lat = [
    (4, "kola", "maaselka", (9189, 7666), (9209, 7598), 35.44, 0.50),
    (9, "kirkenes", "varanger", (9041, 7715), (9010, 7759), 26.91, 0.50),
    (13, "pyalitsa", "ponoi", (9521, 7479), (9520, 7537), 29.00, 0.50),
]

print(f"\n{'ID':<3} | {'Route':<20} | {'PxDist':<8} | {'dx':<5} | {'dy':<5} | {'dx/dy':<6} | {'Actual':<7} | {'Ratio':<6} | Status")
print("-" * 90)

for id, start, end, p1, p2, actual, terrain in working_high_lat:
    px_dist = pixel_distance(p1, p2)
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    dx_dy = dx / dy if dy > 0 else float('inf')
    ratio = actual / px_dist
    print(f"{id:<3} | {start}->{end:<10} | {px_dist:<8.2f} | {dx:<5} | {dy:<5} | {dx_dy:<6.1f} | {actual:<7.2f} | {ratio:<6.3f} | WORKS")

for id, start, end, p1, p2, actual in norway_tests:
    px_dist = pixel_distance(p1, p2)
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    dx_dy = dx / dy if dy > 0 else float('inf')
    ratio = actual / px_dist
    print(f"{id:<3} | {start}->{end:<10} | {px_dist:<8.2f} | {dx:<5} | {dy:<5} | {dx_dy:<6.1f} | {actual:<7.2f} | {ratio:<6.3f} | OUTLIER")
