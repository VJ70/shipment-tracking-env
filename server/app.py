from openenv import OpenEnv

app = OpenEnv()

@app.tool()
def track_shipment(tracking_id: str) -> dict:
    """Track a shipment by its ID."""
    return {
        "tracking_id": tracking_id,
        "status": "in_transit",
        "location": "Mumbai Hub",
        "eta": "2026-04-10"
    }

def main():
    app.run()

if __name__ == "__main__":
    main()