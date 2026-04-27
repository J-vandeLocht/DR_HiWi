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

            outputs = model(inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            prob = torch.sigmoid(logits).item()

            results[video_name] = {
                "label": int(labels.item()),
                "prob": prob
            }
    return results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "predictions_master.csv")

    # --- Setup Data ---
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs)
    val_ds = MILVideoDatasetNew(
        json_path="data/mil_val.json",
        num_frames=32,
        transform=val_trans
    )
    # shuffle MUST be False for our index-based name retrieval to work
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

    # Load/Initialize Master CSV
    if os.path.exists(csv_path):
        master_df = pd.read_csv(csv_path, index_col='video')
        print(f"Resuming from existing master file.")
    else:
        master_df = pd.DataFrame()

    # --- Loop over models ---
    for model_path in args.model_paths:
        # Create unique column name: parent_folder + filename
        parent_dir = os.path.basename(os.path.dirname(model_path))
        file_base = os.path.basename(model_path).replace(".pth", "")
        run_name = f"{parent_dir}_{file_base}"

        if run_name in master_df.columns:
            print(f"Skipping {run_name}, already processed.")
            continue

        print(f"\n>>> Evaluating Run: {run_name}")

        model = DinoMIL(
            repo_dir=args.repo_dir,
            weights=args.weight_path,
            checkpoint_path=None,
            freeze_backbone=False,
            use_cls=args.use_cls
        ).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))

        run_results = run_inference(model, val_loader, device)

        # Convert to temp DF
        temp_df = pd.DataFrame.from_dict(run_results, orient='index')
        temp_df.index.name = 'video'

        if master_df.empty:
            # First run establishes the 'label' and first 'prob' column
            master_df = temp_df.rename(columns={'prob': run_name})
        else:
            # Join subsequent runs on video name index
            master_df[run_name] = temp_df['prob']

        # Save after every successful model to prevent data loss
        master_df.to_csv(csv_path)

    print(f"\nAll 4 runs completed. Results: {csv_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--repo_dir', type=str, required=True)
    parser.add_argument('--weight_path', type=str, required=True)
    parser.add_argument('--model_paths', nargs='+', required=True,
                        help='Provide all 4 .pth paths here')

    parser.add_argument('--output_dir', type=str, default="mil_multi_eval")
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--complex_augs', action='store_true')
    parser.add_argument('--use_cls', action='store_true')

    args = parser.parse_args()
    main(args)