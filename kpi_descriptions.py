# Create the enhanced kpi_descriptions structure with additional fields: description, unit, and normal_range

kpi_descriptions = {
    "dew_point_2m": {
        "description": "Temperature at which air reaches saturation (humidity marker)",
        "unit": "°C",
        "normal_range": "10–20°C"
    },
    "wet_bulb_temperature_2m": {
        "description": "Temperature accounting for evaporation cooling effect",
        "unit": "°C",
        "normal_range": "10–25°C"
    },
    "temperature_2m": {
        "description": "Standard air temperature at 2 meters above surface",
        "unit": "°C",
        "normal_range": "10–30°C"
    },
    "temperature_2m_mean": {
        "description": "Average daily air temperature at 2 meters",
        "unit": "°C",
        "normal_range": "15–25°C"
    },
    "relative_humidity_2m": {
        "description": "Percentage of moisture in air compared to max capacity",
        "unit": "%",
        "normal_range": "40–70%"
    },
    "cloud_cover": {
        "description": "Fraction of sky obscured by clouds",
        "unit": "%",
        "normal_range": "20–80%"
    },
    "cloud_cover_high": {
        "description": "Coverage by high-altitude clouds",
        "unit": "%",
        "normal_range": "10–50%"
    },
    "precipitation_sum": {
        "description": "Total precipitation over a period",
        "unit": "mm",
        "normal_range": "0–10 mm/day"
    },
    "total_column_integrated_water_vapour": {
        "description": "Vertical total of atmospheric water vapor",
        "unit": "kg/m²",
        "normal_range": "10–50 kg/m²"
    },
    "et0_fao_evapotranspiration": {
        "description": "FAO Penman-Monteith evapotranspiration estimate",
        "unit": "mm/day",
        "normal_range": "2–6 mm/day"
    },
    "pressure_msl": {
        "description": "Atmospheric pressure at mean sea level",
        "unit": "hPa",
        "normal_range": "1000–1025 hPa"
    },
    "surface_pressure": {
        "description": "Actual pressure at the Earth's surface",
        "unit": "hPa",
        "normal_range": "950–1050 hPa"
    },
    "wind_speed_10m": {
        "description": "Horizontal wind speed at 10 meters",
        "unit": "m/s",
        "normal_range": "1–6 m/s"
    },
    "wind_direction_10m": {
        "description": "Wind direction (degrees) at 10 meters",
        "unit": "degrees",
        "normal_range": "0–360°"
    },
    "apparent_temperature": {
        "description": "Feels-like temperature combining wind & humidity",
        "unit": "°C",
        "normal_range": "10–30°C"
    },
    "soil_temperature_0_to_7cm": {
        "description": "Top-layer soil temperature",
        "unit": "°C",
        "normal_range": "10–25°C"
    },
    "soil_moisture_0_to_7cm": {
        "description": "Top-layer soil moisture content",
        "unit": "m³/m³",
        "normal_range": "0.1–0.4 m³/m³"
    },
    "rain": {
        "description": "Rainfall depth",
        "unit": "mm",
        "normal_range": "0–15 mm/day"
    },
    "snowfall": {
        "description": "Accumulated snowfall",
        "unit": "mm",
        "normal_range": "0–20 mm/day"
    },
    "vapour_pressure_deficit": {
        "description": "Drying power of air – gap between vapor pressure & saturation",
        "unit": "kPa",
        "normal_range": "0.5–2.5 kPa"
    },
    "shortwave_radiation": {
        "description": "Solar radiation reaching the surface (W/m²)",
        "unit": "W/m²",
        "normal_range": "100–800 W/m²"
    },
    "sunshine_duration": {
        "description": "Hours of sunlight per day",
        "unit": "hours",
        "normal_range": "5–14 hours"
    },
    "wine_quality_score": {
        "description": "Predicted quality score of wine based on features",
        "unit": "score (0–10)",
        "normal_range": "4–8"
    }
}
