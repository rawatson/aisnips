**TL;DR**: The market access formulas make land <=> sea transitions *way* too cheap and the distance calculations seriously disadvantage northern markets

# Background: market access and map projections

Market access matters a lot in-game.  A basic spinners guild with 50% market access will consume 0.5 wool and produce 0.5 cloth (instead of 1 wool => 1 cloth).  This means low market access limits profits and hurts your ability to meet the needs of your pops.  Given the massive impact market access has, it's surprisingly poorly explained in-game.  There are in-game tooltips for things like:

* Traversing up / down a river improves market access
* Building marketplace or market village buildings in a location improves market access
* Harbor capacity impacts market access
* If either side is an ocean sea tile, add an additional 50% penalty
* Building roads / railroads improves market access

[This post](https://www.reddit.com/r/EU5/comments/1oyq07w/market_access_cost_explained/) gives a good walkthrough of these in-game effects (though there are [some bugs in the tooltips](https://forum.paradoxplaza.com/forum/threads/multiple-issue-with-market-access-calculation-tooltip.1884559/)).  However, these don't explain a critical factor: where do the original values for "Terrain distance between x and y" (image #1) come from?  To answer this we need to first explain the map projection.

In [Tinto Talk #2](https://forum.paradoxplaza.com/forum/developer-diary/tinto-talks-2-march-6th-2024.1626415/), the developers confirmed they were using [Gall stereographic projection](https://en.wikipedia.org/wiki/Gall_stereographic_projection) to represent the world.  Under the hood, this means they use a 16384 × 8192 rectangular grid, where the equator is placed at y = 3340, and x = 0 is a little bit to the east of Samoa.  You can see the coordinates of any location by opening up debug mode, then looking at the "Origo" coordinates, which correspond to the "center" of a location (see image #2 for an example near the equator).

# Calculating terrain distances

Also in image #2 you can see more data beyond the coordinates, including that luuq => hudur has a terrain distance of 22.58.  Market access loss is simply half of that value, so traversing from Luuq to Hudur costs 11.29% market access -- which is confirmed in image #1.

OK, but where the heck did the game get that 22.58 number from?  It's based on a combination of (1) the distance between the Origo points of each location and (2) a terrain multiplier.  Let's work through an example to illustrate.

First, we compute the distance by finding the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) between the location's Origo points.   Luuq is at 9646, 3583, and Hudur is at 9698, 3561, so the distance is:

    distance = sqrt((9646 - 9698)² + (3583 - 3561)²)
             = sqrt(3188)
             ≈ 56.46

Next, we compute the terrain multiplier for each location:

    | Topography | Base Cost |  | Vegetation | Modifier |
    |------------|-----------|  |------------|----------|
    | flatland   | 0.40      |  | sparse     | +0.00    |
    | ocean      | 0.40      |  | grasslands | +0.00    |
    | plateau    | 0.45      |  | desert     | +0.00    |
    | hills      | 0.50      |  | farmland   | +0.02    |
    | wetlands   | 0.50      |  | woods      | +0.05    |
    | mountains  | 0.60      |  | forest     | +0.10    |
    |            |           |  | jungle     | +0.10    |

Luuq is flatland / sparse vegetation, and Hudur is flat grasslands, so they both have a multiplier of 0.40.  We average the adjusted terrain costs for a final formula of:

    cost = distance * avg(terrain_mult₁, terrain_mult₂)
         = 56.46 * avg(0.40, 0.40)
         = 56.46 * 0.40
         = 22.58

Just what it said in the in-game UI!  You can repeat this process to verify other locations costs in game.  This was a pain to reverse engineer, but I automated testing of this for almost 70 different location pairs and the formula worked for all of them.

Unfortunately, the approach Paradox chose has two major issues that I'll note here:

# Issue #1: land => sea transitions are way too cheap

When going from land to sea, the game always uses a fixed, absurdly good, terrain multiplier of 0.2, which is 2-3x as efficient as land / sea alone.  Even having zero harbor access, only gives a 1% market access penalty (or 5% if entering from a non-port sea location), which is negligible compared to the value of a 0.2 terrain multiplier in most cases.

In practice, this means the game will take as many land => sea trips as possible.  Take the Aegean islands of [Sporades](https://en.wikipedia.org/wiki/Sporades) as an example (see image #3 for an illustration of the path described here).  Logically, one would expect to leave the island, then sail directly Cape Lecton => Dardanelles => Sea of Marmara and land at Constantinople.  Super easy, right?  Instead, the game will:

1. Enter the sea and sail to Cape Lecton
2. Despite zero harbor access, land at Ayvacik and walk to Canakkale
3. Get back on the boats into the Dardanelles.
4. Despite miniscule harbor access, land at Gonen
5. Get on a boat again in Lake Manyas (!!!)
6. Immediately get off the boat and land at Bandirma.
7. Get back on a boat into the sea of Marmara and land at Constantinople.

This is comical.  Instead of just sailing between Cape Lecton and the Dardanelles, we instead: (1) disembark from Cape Lecton, (2) walk across a province, (3) get back on our boats in the Dardanelles!  Only to repeat this exact same tactic using *a scenic boat ride on Lake Manyas instead of just walking.*  This is the mathematically correct shortest path to Constantinople in the game's view, but the result is logically absurd.

This also helps explain complaints about [market access sea costs being too high](https://www.reddit.com/r/EU5/comments/1pn99ec/market_access_between_sea_tiles_needs_to_be/).  It's not just that the sea is expensive, it's also that going back / forth between land and sea is absurdly cheap.

# Issue #2: distance is significantly distorted farther from the equator

The other problem is that the game doesn't correct for the spherical nature of the world when computing distances.  You can visualize this by picturing the surface of a globe "stretching out" onto a flat table -- you're going to have to really stretch out the far northern and southern parts of the map in order to make a rectangular shape.  This introduces significant distortion for east <=> west moves at northern latitudes like Scandinavia.  In order to get the "true" distance, you need to transform from Gall stereographic coordinates back to [great circle](https://en.wikipedia.org/wiki/Great_circle) coordinates.  I'll spare you the math behind the calculation here, but I'll refer to this as the "true distance" in km and show an example.

Consider the move from Luuq to Hudur that I mentioned in the background section / image #1 and #2.  In this case:

    luuq  = 9646, 3583
    hudur = 9698, 3561
    
    euclidean distance = 56.5
    true distance = 134km

OK, that sounds fine.  Let's consider an example from the far north of Norway, going from Vardo to Deatnu.

    vardo  = 9049, 7794
    deatnu = 8968, 7801
    
    euclidean distance = 81.3
    true distance = 66km

So despite the fact that the distance between Vardo / Deatnu is *less than half* of the Luuq / Hudur distance, the game treats it as if its a >40% *longer* route?  The horizontal pixels / km ratio in this case is almost **three times worse** than at the equator.  If the map had extended all the way to [Svalbard ](https://en.wikipedia.org/wiki/Svalbard)this distortion could be up to 5x worse.  This makes it increasingly difficult to build a healthy market the farther north you go.

# What should be done about this?

* There are a lot of different ways to solve #1.  The game could make sea / land more expensive, increase harbor access costs, or make sea routes cheaper than 0.4.  Each of these changes will have spillover effects, so the devs will likely have to test a few different options.  Personally, I'd start by testing a land <=> sea terrain modifier of 0.4 then tuning the impact of harbor access to produce a reasonable looking map, but that's just one approach.
* Solving issue #2 is straightforward conceptually but will take a bit more work mathematically.  The game needs to apply the inverse projection and [haversine formula](https://en.wikipedia.org/wiki/Haversine_formula) (or some other trigonometric approximation) when computing distances.  This will have the downside of making market access in the far north seem "suspiciously good" for players judging distances visually using the rectangular in-game map, but it's a more accurate model of the world.  I'll credit [this post](https://forum.paradoxplaza.com/forum/threads/comprehensive-feedback-on-market-access-calculations-its-interplay-with-the-graph-underlying-the-world-map-cf-origo-in-debug-mode-and-waterways.1876397/) on the forums for a dense but helpful discussion of how some of this could be computed for those who are curious.