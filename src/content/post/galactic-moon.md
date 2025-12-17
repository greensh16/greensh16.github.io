---
title: "Galactic Moon: A Moon Tracker on Galactic Unicorn"
description: "Turning a 53×11 LED matrix into a tiny horizon that shows where the Moon is right now."
publishDate: "2025-12-12"
tags: ["micropython", "pico-w", "led", "astronomy", "moon", "hardware"]
---

I like projects that feel a little bit magical when they’re working.

Galactic Moon started as a simple itch: *I wanted to look up and know where the Moon actually was*—not in an app, not on a map, but in something physical I could leave running on my desk.

I already had a **Pimoroni Galactic Unicorn** (a 53×11 RGB LED matrix driven by a Raspberry Pi Pico W), so I turned it into a tiny horizon.

- A single **green pixel** marks “you”, facing **north**.
- The Moon becomes **four white pixels** (a small 2×2 block).
- Its position moves across the matrix as the Moon moves across the sky.

## The core idea: azimuth + altitude → pixels
Astronomy software can get very complicated very quickly, but for a desk display you don’t need much.

If you know:

- **Altitude** (how high above the horizon)
- **Azimuth** (compass direction, 0–360°)

…then you can plot a point in a sky map.

On the Galactic Unicorn, that “sky map” is just 53 pixels wide by 11 pixels tall.

### Mapping altitude
Altitude is the easy one:

- Valid range: **0° (horizon)** to **90° (straight up)**
- Screen range: **y = 10** (bottom row) to **y = 0** (top row)

So the code normalizes altitude into that 0–10 pixel span.

### Mapping azimuth (and why it’s location-dependent)
Azimuth is trickier because a tiny display can’t show *all* directions equally well.

In `main.py` I picked two extents that make sense for my location in the **southern hemisphere**:

- `az_east = 125°`
- `az_west = 235°`

Those are treated as “the visible slice of sky” across the width of the display.

If you’re in a different region (especially the northern hemisphere), you’ll want to adjust:

1. The azimuth extents near the top of `main.py`
2. Any azimuth correction logic in the Moon calculation (see notes in `moon.py`)

## The math: getting the Moon’s position without an API
One of my favorite parts of this project is that it doesn’t depend on a web service for ephemerides.

The Pico W connects to Wi‑Fi only to set an accurate clock via NTP (`ntptime.settime()`), then everything else is local math.

The `MoonPosition` class in `moon.py` computes the Moon’s azimuth/altitude using a lightweight set of astronomical formulas (inspired by Vladimir Agafonkin’s `suncalc` and other references).

At a high level, the flow is:

1. Convert a Unix timestamp → **days since J2000** (using a Julian date conversion)
2. Compute the Moon’s **mean longitude**, **mean anomaly**, and **mean distance from the ascending node**
3. Apply small **perturbation** terms to improve accuracy
4. Convert to **right ascension** and **declination**
5. Use local **sidereal time** to compute the hour angle
6. Convert that into **altitude** and **azimuth** for your latitude/longitude

It also returns an approximate Earth–Moon distance in km, which could be a fun future hook (e.g. brightness scaling).

## Timing + display loop
Once the Pico has a correct clock, the main loop is intentionally straightforward:

- Clear the display
- Draw the north marker (green pixel)
- Compute the Moon’s position for the current time + your lat/lon
- Draw the Moon (2×2 white pixels)
- Sleep for **10 minutes** and repeat

There’s also a small overclock (`machine.freq(200_000_000)`) so the device stays snappy.

## Setup
High-level setup looks like this:

1. Flash the Galactic Unicorn MicroPython firmware to the Pico W.
2. Copy the project’s `.py` files onto the device.
3. Create a `secrets.py` containing Wi‑Fi credentials and your location.

Example `secrets.py` (names match what the code imports):

```python
ssid = "YOUR_WIFI_SSID"
password = "YOUR_WIFI_PASSWORD"
latitude = -33.8688
longitude = 151.2093
```

## Where I’d like to take it next
The current version is deliberately minimal, but it’s set up nicely for upgrades:

- Track the **Sun**
- Represent **Moon phase / brightness**
- Track a few **planets**
- Change the **background colour** based on time of day

## Source
The code is here:

- `https://github.com/greensh16/Galactic-Moon`

The next time I revisit this project, I want to add just one more layer of “feel”: a background gradient that slowly shifts from day to night while the Moon slides across it.