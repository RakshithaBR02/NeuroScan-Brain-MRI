import torch
import os

MODEL_DIR = "models"

files = [
    "mdunet_figshare_best.pth",
    "swinct_classifier_best.pth",
    "hcac_classifier_corrected.pth"
]

for filename in files:
    print("\n" + "=" * 65)
    print(filename)
    print("=" * 65)

    checkpoint = torch.load(
        os.path.join(MODEL_DIR, filename),
        map_location="cpu",
        weights_only=False
    )

    state = checkpoint["model_state_dict"]

    for name, tensor in state.items():

        # Print all MDU-Net and HCAC layers.
        if filename.startswith("mdunet") or filename.startswith("hcac"):
            print(name, tuple(tensor.shape))

        # For SwinCT, print custom layers and classifier,
        # excluding the internal Swin Transformer layers.
        elif ".swin." not in name:
            print(name, tuple(tensor.shape))

print("\nARCHITECTURE INSPECTION COMPLETE!")