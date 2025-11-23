
DEFAULT_WEATHER_RESPONSE = {
    "weather": [{"main": "Clear", "description": "clear sky"}],
    "main": {"temp": 25, "feels_like": 26, "temp_min": 24, "temp_max": 26, "pressure": 1012, "humidity": 60},
    "wind": {"speed": 3.1},
    "sys": {"country": "US"},
    "name": "Test City",
    "id": 12345,
}


MOCK_BRAVE_SEARCH_API_RESPONSE = {
    "web": {
        "results": [
            {
                "title": "SpaceX Sets Record with Most Falcon 9 Launches in a Year",
                "url": "https://www.space.com/spacex-falcon-9-launch-record-2025",
                "description": "SpaceX has broken its own record for the most Falcon 9 launches in a single year, with the 100th launch of 2025 carrying another batch of Starlink satellites.",
                "page_age": "2025-11-20T12:00:00Z",
            },
            {
                "title": "NASA's Artemis II Mission Crew Prepares for Lunar Flyby",
                "url": "https://www.nasa.gov/artemis-ii-crew-preparation",
                "description": "The crew of the Artemis II mission are in their final phase of training for the historic lunar flyby, scheduled for next month.",
            },
        ]
    }
}

MOCK_WEBPAGE_CONTENT = """
<!DOCTYPE html>
<html>
<head><title>SpaceX Launch Record</title></head>
<body>
    <h1>SpaceX Smashes Launch Record in 2025</h1>
    <p>In a historic achievement, SpaceX has successfully completed its 100th Falcon 9 launch of 2025. This new record underscores the company's dominance in the commercial spaceflight industry.</p>
    <p>The mission, designated Starlink 10-2, lifted off from Cape Canaveral Space Force Station.</p>
</body>
</html>
"""
