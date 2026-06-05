import torch
import numpy as np
import pandas as pd
import os
import re
from tqdm import tqdm
from data.dataset import MILVideoDatasetNew
from .utils import make_val_transform_dino
from .train_dino_mil import DinoMIL


def run_inference(model, loader, device):
    """
    Inference loop that pulls video names directly from the dataset
    object to avoid indexing errors in the DataLoader batch.
    """
    model.eval()
    results = {}

    # We access the internal 'data' list of MILVideoDatasetNew
    dataset_samples = loader.dataset.data

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Running Inference")):
            # loader returns (bag, label), batch[0] is bag
            inputs = batch[0].squeeze(0).to(device)
            labels = batch[1]

            # Direct access to the dataset info via index
            # This is safe because shuffle=False in main()
            video_name = dataset_samples[i]['vid_name']

            logits, _ = model(inputs)
            prob = torch.sigmoid(logits).item()

            results[video_name] = {
                "label": int(labels.item()),
                "prob": prob
            }
    return results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs)
    # for split in [1, 2, 3, 4, 5]:
    for split in [5]:
        csv_path = os.path.join(args.output_dir, f"predictions_split_{split}.csv")

        # --- Setup Data ---
        val_ds = MILVideoDatasetNew(
            json_path=f"data/stratified_splits/split_{split}/mil_val.json",
            num_frames=32,
            transform=val_trans
        )
        # shuffle MUST be False for our index-based name retrieval to work
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

        run_name = f"split_{split}"

        print(f"\n>>> Evaluating Run: {run_name}")

        # model = DinoMIL(
        #     checkpoint_path=args.model_paths[split - 1]
        # ).to(device)
        model = DinoMIL(
            checkpoint_path=args.model_paths[0]
        ).to(device)

        run_results = run_inference(model, val_loader, device)

        # Convert to temp DF
        temp_df = pd.DataFrame.from_dict(run_results, orient='index')
        temp_df.index.name = 'video'

        master_df = temp_df.rename(columns={'prob': run_name})
        master_df.to_csv(csv_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_paths', nargs='+', required=True)
    parser.add_argument('--output_dir', type=str, default="mil_multi_eval")
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--complex_augs', action='store_true')

    args = parser.parse_args()
    main(args)
