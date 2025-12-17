---
title: "ClimTime: Real-Time Weather Timeline on Galactic Unicorn"
description: "A tiny MicroPython project that turns two days of weather into a scrolling LED tapestry."
publishDate: "2025-12-12"
tags: ["micropython", "pico-w", "led", "weather", "visualization", "hardware"]
---

I built ClimTime because I wanted a weather display that felt like *weather*.

Not “it’s 23°C and 40% cloudy”—but the shape of the last couple of days: a bright patch of clear sky, a windy afternoon, a short burst of rain, the way temperature ramps up (or doesn’t) as daylight arrives.

Sitting on my desk was a **Pimoroni Galactic Unicorn**: a 53×11 RGB LED matrix driven by a Raspberry Pi Pico W. It was the perfect canvas for something that could be read at a glance.

So ClimTime became a simple idea:

- **each column is one hour**
- **53 columns ≈ a little over two days**
- the whole display becomes a scrolling timeline you can *feel* more than you can “read”

## From numbers to texture
The heart of ClimTime is a mapping from a few weather signals into light:

- **Temperature → color (HSV hue ramp)**
  - I clamp temperature to a sensible range (about **−10°C to 40°C**).
  - Hot hours trend toward **red**; cold hours trend toward **blue**.
  - Implementation detail: a hue value from `0.0` (red) to ~`0.66` (blue), converted via a tiny `hsv_to_rgb()`.

- **Cloud cover → brightness**
  - Clear sky becomes brighter; overcast becomes dimmer.
  - There’s a floor (about **20% brightness**) so the column never disappears completely.

- **Wind speed → bar height**
  - Windier hours draw taller bars.
  - Implementation detail: clamp wind to **0–50 kph**, map it into **1–11 pixels**.

- **Precipitation → sparkle**
  - If precip is “meaningful” (threshold ~**0.5 mm**), the top pixel flashes white.

All of that logic lives in `map_weather.py`, which takes a small weather dict and returns one 11‑pixel column.

## The part I’m happiest with: it has memory
A single “current conditions” display is easy. The fun part is giving it history.

On boot, ClimTime backfills the timeline by fetching hourly observations for the last ~53 hours (limited by the 53‑pixel width):

1. Compute timestamps for each of the last 53 hours.
2. Group them by date.
3. Fetch hourly history for each date.
4. Build the initial list of 53 columns, then start appending “now”.

After that, it’s a rhythm:

- fetch the latest weather
- map it into a new column
- shift left and append
- redraw the whole frame
- sleep until the next hour boundary

That loop is intentionally blunt—no complicated incremental rendering—because on a 53×11 canvas, redrawing everything is cheap and keeps the code readable.

## Where the data comes from
ClimTime pulls JSON from **WeatherAPI** using MicroPython’s `urequests`.

- `weather.py` fetches **current conditions** (for the latest column) and **hourly history** (to backfill the timeline on boot).
- I keep only the handful of fields the renderer cares about: temperature, cloud cover, wind speed, precipitation, and a day/night flag.

That “small dict in, single column out” approach keeps the rest of the code simple.

## A quiet background that hints at day and night
One subtle touch I like: the weather bars are drawn *over* a background wash that changes by time of day.

Night is very dim, day is brighter, and sunrise/sunset get a warmer tint. It gives the timeline structure, even before the weather data is layered on top.

## Timekeeping (MicroPython reality)
To keep the hourly ticks honest, ClimTime syncs the Pico W’s RTC via NTP (`ntptime.settime()`), then applies a fixed `UTC_OFFSET` to produce local time.

If you’re in a different timezone, adjusting `UTC_OFFSET` in `rtc_time.py` is the first edit.

## Hardware + setup
You’ll need:

- Pimoroni Galactic Unicorn (includes a Raspberry Pi Pico W)
- USB power (a stable 5V supply is worth it)

High-level setup:

1. Flash the Galactic Unicorn MicroPython firmware to the Pico W.
2. Copy the project files to the Pico.
3. Add a `secrets.py` with Wi‑Fi + API credentials.
4. Run `main.py`.

Example `secrets.py`:

```python
WIFI_SSID = "YOUR_SSID"
WIFI_PASSWORD = "YOUR_PASSWORD"
API_KEY = "YOUR_WEATHER_API_KEY"
LOCATION = "Sydney"
```

## Where I’d like to take it next
ClimTime is intentionally small, but it opens up a bunch of good next steps:

- a true **7‑day scroll mode** (168 hours)
- weather “events” (lightning flashes, storm intensity)
- auto‑dimming using the onboard light sensor
- DST‑aware timezone handling
- local sensor fallback when Wi‑Fi drops
- a tiny UI via the Unicorn’s buttons

## Source
The code is on GitHub:

- `https://github.com/greensh16/ClimTime`

The next upgrade I want is the least technical one: a photo or short video of the display after a full day of real weather. The whole point of ClimTime is that it’s hard to describe—and easy to *see*.
