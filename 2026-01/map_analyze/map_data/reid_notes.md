Map is 16384x8192, with the equator at y = 3340.  Uses Gall stereographic projection

# First test: Near equator in ajuraan, hudur => luuq
hudur (ID 17167): 9698, 3561
luuq (ID 17178): 9646, 3583

Pixel dist = 56.46
In game distance =  22.58 <== ???

# Second test: land => land, far north
Deatnu: 8698, 7801
Vardo: 9049, 7794

Pixel dist = 351.13
In game distance = 40.65 <== ???

# Third test: Byzantine, constantinople => silivri
constantinople (ID 5872): 9005, 5709
silivri (ID 5875): 8981, 5710

Pixel dist: 24.02
In game distance: 10.08 <== ???

# Fourth test: far north, land => land
kola: 9189, 7666
maaselka: 9209, 7598

Pixel dist = 70.88
In game distance = 35.44 <== pixel distance / 2

# Fifth test: land => land
loimola: 9125, 7118
salmi: 9141, 7101

Pixel dist: 23.34
In game distance = 11.67 <== pixel distance / 2

# Sixth test: note that this is land => lake
loimola: 9125, 7118
lake_ladoga: 9122, 7043

Pixel dist: 75.05
In game distance = 15.01 <== pixel distance / 5

# Seventh test: far north: inland sea => inland sea
intsi_cape: 9461, 7394
varzuga_estuary: 9273, 7447

Pixel dist: 192.86
In game distance = 78.13 < == ???

# Eigth test: inland sea => inland sea
inner_onega_bay: 9352, 7319
varzuga_estuary: 9273, 7447

Pixel distance: 150.41
In game distance = 60.16 <== pixel distance / 2.5

#Ninth test: land => land, far north
kirkenes: 9041, 7715
varanger: 9010, 7759

Pixel dist: 53.82
In game distance: 26.91 <== pixel distance / 2

#Tenth test: land => lake, in Byzantine area
gonen: 8952, 5647
lake_manyas: 8964, 5650

Pixel dist: 12.37
In game dist: 2.47 <== pixel distance / 5

# Eleventh test: land => Lake, in Byzantine area
lake_manyas: 8964, 5650
bandirma: 8967, 5662

Pixel dist: 12.37
In game dist: 2.47 <== pixel distance / 5

# Twelfth test: land => land, in Byzantine area
gonen: 8952, 5647
bandirma: 8967, 5662

Pixel dist: 21.21
In game dist: 11.13 <== ???

# Thirteenth test: land => land, far north
pyalitsa: 9521, 7479
ponoi: 9520, 7537

Pixel dist: 58.01
In game dist: 29.00 <== pixel distance / 2

# Fourteenth test: sea => sea, near equator
cadde_cape: 9808, 3472
banaadir_coast: 9762, 3433

Pixel dist: 60.31
In game dist: 24.12 <== pixel distance / 2.48

# Fifteenth test: land => land, further south
great_zimbabwe: 9095, 2239
mabveni: 9082, 2252

Pixel dist: 18.44
In-game dist: 10.11 <== ???

# Sixteenth test: land => land, southernmost SA
juni_aiken: 4478, 296
ciaike: 4516, 264

Pixel dist: 49.477
In-game dist: 19.87 <== pixel dist / 2.49

# Seventeenth test: land => land, southernmost SA
nakenk: 4619, 126
hamenk: 4587, 136

Pixel dist: 33.53
In-game dist: 15.08 <== ???

# Eighteenth test: sea => sea, southernmost SA
aguirre_bay: 4693, 42
blossom_bay: 4794, 69

Pixel dist: 104.38
In-game dist: 41.81 <== pixel dist / 2.50

#Nineteenth test: land => land, southernmost SA
utumaala: 4627, 73
wakimaala: 4593, 76

Pixel dist: 34.13
In-game dist: 20.47 <== pixel_dist / ???

# 20th test: land => land, southernmost SA
hamenk: 4587, 136
kauwes: 4554, 142

Pixel dist: 33.54
In-game dist: 15.09 <== ???

#21st test: land => land, southern SA
corpen_aike: 4526, 441
chalten: 4437, 439

Pixel dist: 89.02
In-game dist: 40.06