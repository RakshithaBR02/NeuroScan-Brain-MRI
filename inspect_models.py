import torch
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model_files = [
    "mdunet_figshare_best.pth",
    "swinct_classifier_best.pth",
    "hcac_classifier_corrected.pth"
]

for filename in model_files:
    path = os.path.join(MODEL_DIR, filename)

    print("\n" + "=" * 60)
    print("MODEL:", filename)
    print("=" * 60)

    if not os.path.exists(path):
        print("ERROR: File not found!")
        continue

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(checkpoint, dict):
        print("Checkpoint keys:", list(checkpoint.keys()))

        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint)
        )

        if isinstance(state_dict, dict):
            print("\nTotal parameter tensors:", len(state_dict))

            print("\nFirst 15 parameter names and shapes:")
            for i, (key, value) in enumerate(state_dict.items()):
                if i == 15:
                    break
                if hasattr(value, "shape"):
                    print(key, tuple(value.shape))
    else:
        print("Checkpoint type:", type(checkpoint))

print("\n\nMODEL INSPECTION COMPLETED!")