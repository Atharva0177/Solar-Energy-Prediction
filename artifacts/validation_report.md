# Data Quality Report (auto-generated)

_Post-cleaning state. Timezone used for solar position: **Australia/Melbourne**._

- Final merged rows: 2,731,946
- Final columns: `timestamp, campus_id, site_id, power, solar_elevation_deg, is_daylight, apparent_temperature, temperature, dew_point_temperature, humidity, wind_speed, wind_direction, year, month`

## generation

- Duplicate keys (site_id+timestamp): **0**
- Missing timestamps (15min): **52492** slots across 42 groups
  - worst: site_id=13 missing 2.157% (1704/79007 slots)
  - worst: site_id=1 missing 2.103% (1704/81023 slots)
  - worst: site_id=2 missing 2.103% (1704/81023 slots)
- Missing % by variable: `power`=56.235
- Impossible values: none detected

## weather

- Duplicate keys (campus_id+timestamp): **0**
- Missing timestamps (15min): **0** slots across 5 groups
  - worst: campus_id=1 missing 0.0% (0/81017 slots)
  - worst: campus_id=2 missing 0.0% (0/81017 slots)
  - worst: campus_id=3 missing 0.0% (0/81017 slots)
- Missing % by variable: `apparent_temperature`=28.812, `temperature`=28.812, `dew_point_temperature`=28.812, `humidity`=28.812, `wind_speed`=43.815, `wind_direction`=43.815
- Impossible values: none detected
