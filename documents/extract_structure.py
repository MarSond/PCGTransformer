import json
from pprint import pprint
import os

def load_and_summarize_json(file_path):
    file_path = os.path.abspath(os.path.normpath(file_path))
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}")
        return None
    except Exception as e:
        print(f"Error opening or reading file: {file_path}")
        print(f"Error: {str(e)}")
        return None

    summary = {
        "fold_data": {
            "training": {},
            "validation": {}
        },
        "averages": {
            "training": {},
            "validation": {}
        }
    }
    
    # Summarize fold_data
    for mode in ["training", "validation"]:
        if mode in data.get("fold_data", {}):
            first_fold = next(iter(data["fold_data"][mode].values()), None)
            if first_fold:
                first_epoch = first_fold[0]
                summary["fold_data"][mode] = {
                    "structure": {k: type(v).__name__ for k, v in first_epoch.items()},
                    "example": {k: v for k, v in first_epoch.items() if k not in ["curve_data", "confusion"]}
                }
                if "curve_data" in first_epoch:
                    summary["fold_data"][mode]["curve_data"] = {
                        k: type(v).__name__ for k, v in first_epoch["curve_data"][0].items()
                    }
    
    # Summarize averages
    for mode in ["training", "validation"]:
        if mode in data.get("averages", {}):
            for metric, values in data["averages"][mode].items():
                if values:
                    summary["averages"][mode][metric] = {
                        "structure": type(values[0]).__name__,
                        "example": values[0] if metric not in ["curve_data", "confusion"] else "..."
                    }
    
    return summary

# Usage
file_path = r'E:\Work\PCGClassification\PCGTransformer\runs\2024-06-26_10-43-25_QODJ\metrics.json'
summary = load_and_summarize_json(file_path)
if summary:
    pprint(summary, depth=6)
else:
    print("Failed to load and summarize the JSON file.")