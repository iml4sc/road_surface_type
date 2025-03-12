#!/usr/bin/env python
"""
A script to convert a CSV file (with columns like image_path, Longitude, Latitude, Timestamp, etc.)
into a single JSON file containing a list of GeoPose objects.

Example usage:
  python csv_to_json.py --csv-path TEST_all_3type.csv --output-json input_geopose.json
"""

import argparse
import json
import pandas as pd
from datetime import datetime

def create_geopose_json_item(longitude, latitude, image_url, valid_time=None):
    """
    Create a minimal GeoPose-like JSON structure for a single record.
    You can expand or modify fields according to the OGC GeoPose 1.0 specification.
    """
    if valid_time is None:
        # Use current timestamp in milliseconds if not provided
        valid_time = int(datetime.now().timestamp() * 1000)

    geopose_data = {
        "header": {
            "transitionMode": {
                "authority": "/geopose/1.0",
                "id": "interpolate",
                "parameters": ""
            }
        },
        "outerFrame": {
            "authority": "/geopose/1.0",
            "id": "LTP-ENU",
            "parameters": {
                "longitude": float(longitude),
                "latitude": float(latitude),
                "height": 0.0
            }
        },
        "streamElements": [
            {
                "frame": {
                    "authority": "/geopose/1.0",
                    "id": "RotateTranslate",
                    "parameters": {
                        "translation": [0.0, 0.0, 0.0],
                        "rotation": [0.0, 0.0, 0.0, 1.0]
                    }
                },
                "validTime": valid_time,
                "media": {
                    "image_url": image_url
                }
            }
        ]
    }
    return geopose_data


def convert_csv_to_geopose_json(csv_path, output_json):
    """
    Reads CSV and writes a JSON file containing a list of GeoPose objects.
    CSV columns expected: image_path, Longitude, Latitude, [Timestamp]
    """
    df = pd.read_csv(csv_path)
    geopose_list = []

    for _, row in df.iterrows():
        image_url = row["image_path"]
        longitude = row["Longitude"]
        latitude = row["Latitude"]
        valid_time = row["Timestamp"] if "Timestamp" in df.columns else None

        geo_json_item = create_geopose_json_item(
            longitude=longitude,
            latitude=latitude,
            image_url=image_url,
            valid_time=valid_time
        )
        geopose_list.append(geo_json_item)

    output_data = {
        "geopose_list": geopose_list
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Successfully wrote {len(geopose_list)} GeoPose items to {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Convert CSV to GeoPose JSON")
    parser.add_argument("--csv-path", type=str, required=True,
                        help="Path to the CSV file")
    parser.add_argument("--output-json", type=str, required=True,
                        help="Path to output JSON file")

    args = parser.parse_args()
    convert_csv_to_geopose_json(args.csv_path, args.output_json)


if __name__ == "__main__":
    main()
