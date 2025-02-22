"""
Module for loading the GeoPose JSON file and returning a list of items to be processed.
"""
import json

def load_geopose_json(json_path):
    """
    Loads a GeoPose JSON file that contains a top-level key 'geopose_list'.
    Returns a list of GeoPose items.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "geopose_list" not in data:
        raise ValueError("Invalid GeoPose JSON format: 'geopose_list' key not found.")
    return data["geopose_list"]
