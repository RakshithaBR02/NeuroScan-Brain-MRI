import json
import os

path = "hcac_metadata.json"

if not os.path.exists(path):
    print("Metadata file not found!")
else:
    with open(path, "r") as f:
        data = json.load(f)

    print("METADATA CONTENTS")
    print("=" * 50)

    for key, value in data.items():
        if isinstance(value, list):
            print(f"{key}: length = {len(value)}")
            if len(value) <= 10:
                print(value)
        else:
            print(f"{key}: {value}")

    print("\nMETADATA CHECK COMPLETE!")