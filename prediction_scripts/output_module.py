"""
Module for saving TDML JSON files to disk.
"""

import os
import json

def save_tdml_json(tdml_data, output_dir, base_filename):
    """
    Save TDML JSON data to output_dir as base_filename.json
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{base_filename}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tdml_data, f, indent=2, ensure_ascii=False)
    return out_path
