import math

# Constants
WIDTH = 16400
HEIGHT = 8200
EQUATOR_Y = 3340  # From definitions.txt
EARTH_RADIUS_KM = 6371.0

# Gall Stereographic Projection Constants
# x = R * lambda / sqrt(2)
# y = R * (1 + sqrt(2)/2) * tan(phi / 2)
# Width covers 2*pi radians (360 degrees)

# Calculate Map Radius R (in pixels)
# Width = (R * 2*pi) / sqrt(2)
# R = (Width * sqrt(2)) / (2 * pi)
R_map = (WIDTH * math.sqrt(2)) / (2 * math.pi)

print(f"Map Width: {WIDTH}")
print(f"Map Height: {HEIGHT}")
print(f"Equator Y (from bottom): {EQUATOR_Y}")
print(f"Calculated Map Radius R: {R_map:.2f} pixels")

# Y Offset calculation
# Assuming y_map = y_proj + EQUATOR_Y
# y_proj = R * (1 + sqrt(2)/2) * tan(phi / 2)
def lat_to_y(phi_radians):
    # phi in radians
    y_proj = R_map * (1 + math.sqrt(2)/2) * math.tan(phi_radians / 2)
    return y_proj + EQUATOR_Y

def y_to_lat(y_pixels):
    y_proj = y_pixels - EQUATOR_Y
    term = y_proj / (R_map * (1 + math.sqrt(2)/2))
    # tan(phi/2) = term
    # phi/2 = atan(term)
    # phi = 2 * atan(term)
    return 2 * math.atan(term)

def lon_to_x(lam_radians):
    # lam in radians (-pi to pi)
    # x centered at WIDTH/2 ? Or wrapping 0 to WIDTH?
    # Usually standard: x corresponds to lambda mapped to 0..Width
    # Let's assume x=0 is -pi, x=Width is pi.
    # x = (lam + pi) / (2*pi) * WIDTH
    # Let's verify with projection formula: x_proj = R * lam / sqrt(2)
    # Range of x_proj is [-R*pi/sqrt(2), R*pi/sqrt(2)] = [-W/2, W/2]
    # So x_map = x_proj + WIDTH/2
    x_proj = R_map * lam_radians / math.sqrt(2)
    return x_proj + (WIDTH / 2)

# Calculate Latitude limits
lat_min_rad = y_to_lat(0)
lat_max_rad = y_to_lat(HEIGHT)
lat_min_deg = math.degrees(lat_min_rad)
lat_max_deg = math.degrees(lat_max_rad)

print(f"Latitude Range: {lat_min_deg:.2f} degrees to {lat_max_deg:.2f} degrees")

# Analysis of Distortion
# We want to compare Map Distance (pixels) vs Real Distance (km)
# Cost = Map Distance.
# "Market Access" efficiency = Real KM per Pixel Cost.
# Higher KM/Pixel means "cheaper" access (you cover more real ground for the same map cost).
# Lower KM/Pixel means "expensive" access.
# Alternatively: Map Pixels per 1000 KM.
# Lower Pixels/1000km -> Shorter map distance -> Better access.

print("\n--- Distortion Analysis ---")
print(f"{'Lat':<10} | {'Pixels/1000km (E-W)':<20} | {'Pixels/1000km (N-S)':<20} | {'Scale Factor (vs Eq)':<20}")

lats_to_check = [0, 15, 30, 45, 60, 75]
results = []

for lat_deg in lats_to_check:
    phi = math.radians(lat_deg)
    
    # 1. East-West Distance
    # 1000 km East-West
    # Real circumference at lat: 2 * pi * R_earth * cos(phi)
    # Delta lambda for 1000 km: (1000 / (R_earth * cos(phi))) radians ... approx for small dist
    # Better: delta_lon = 1000 / (R_earth * cos(phi))  (radians)
    # Map distance dx: 
    # x1 = lon_to_x(0)
    # x2 = lon_to_x(delta_lon)
    # dx = x2 - x1
    
    if math.isclose(math.cos(phi), 0, abs_tol=1e-9):
        ew_pixels = float('inf')
    else:
        delta_lon = 1000.0 / (EARTH_RADIUS_KM * math.cos(phi)) # radians
        # x_proj = R * lam / sqrt(2)
        # dx = R * delta_lon / sqrt(2)
        ew_pixels = R_map * delta_lon / math.sqrt(2)
    
    # 2. North-South Distance
    # 1000 km North-South
    # Delta phi = 1000 / R_earth (radians)
    delta_phi = 1000.0 / EARTH_RADIUS_KM
    
    # Calculate dy around the latitude
    # y1 = lat_to_y(phi - delta_phi/2)
    # y2 = lat_to_y(phi + delta_phi/2)
    # dy = y2 - y1
    y1 = lat_to_y(phi - delta_phi/2)
    y2 = lat_to_y(phi + delta_phi/2)
    ns_pixels = abs(y2 - y1)
    
    # Euclidean average (approximate local scale)
    avg_pixels = (ew_pixels + ns_pixels) / 2
    
    # Compare to Equator
    if lat_deg == 0:
        base_pixels = avg_pixels
        ratio = 1.0
    else:
        ratio = avg_pixels / base_pixels

    print(f"{lat_deg:<10} | {ew_pixels:<20.2f} | {ns_pixels:<20.2f} | {ratio:<20.2f}")
    results.append({
        'lat': lat_deg,
        'ew': ew_pixels,
        'ns': ns_pixels,
        'ratio': ratio
    })

print("\n--- Examples ---")
print("Calculating distance between two points separated by 2000km real distance.")

examples = [
    (0, "Equator"),
    (45, "France/Italy latitude"),
    (60, "Scandinavia latitude"),
    (-30, "South Africa/Australia latitude")
]

for lat_deg, name in examples:
    print(f"\nAt {name} ({lat_deg} deg):")
    phi = math.radians(lat_deg)
    
    # Two points 2000km apart East-West
    dist_km = 2000.0
    delta_lon = dist_km / (EARTH_RADIUS_KM * math.cos(phi))
    
    map_dist = R_map * delta_lon / math.sqrt(2)
    print(f"  Real Distance: {dist_km} km")
    print(f"  Map Distance:  {map_dist:.2f} pixels")
    print(f"  Cost Factor:   {map_dist/dist_km:.4f} pixels/km")
