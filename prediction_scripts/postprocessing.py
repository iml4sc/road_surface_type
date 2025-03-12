"""
Module for converting the inference result into a TDML-like JSON structure.
"""

from datetime import datetime

def create_tdml_json(image_path, longitude, latitude, prediction_label, confidence, valid_time=None):
    """
    Create a minimal OGC TrainingDML-AI JSON structure.
    You can expand or modify fields according to your needs.
    """
    if valid_time is None:
        # If no valid_time, use current time in ISO8601
        valid_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    tdml_data = {
        "TDML_version": "1.0",
        "Dataset": {
            "datasetName": "Road Surface Type Classification",
            "DataItem": {
                "FilePath": image_path,
                "Metadata": {
                    "FrameNumber": 1,
                    "Timestamp": valid_time
                },
                "Predictions": {
                    "Label": prediction_label,
                    "Confidence": round(confidence, 4)
                },
                "GeoData": {
                    "Location": {
                        "Longitude": float(longitude),
                        "Latitude": float(latitude)
                    },
                    "SpatialReference": "EPSG:4326"
                }
            }
        }
    }
    return tdml_data
