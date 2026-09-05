import torch
import pandas as pd
import os
from tqdm import tqdm
import argparse
from pathlib import Path

from data.dataset import MILVideoDataset
from dino.utils import make_val_transform_dino
from dino.train_dino_transformer import DinoSelfAttention
from dino.vis_video_results import run_inference_transformer_all


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs)

    for split in [1, 2, 3, 4, 5]:
        csv_path = os.path.join(args.output_dir, f"predictions_split_{split}_{args.dataset_name}.csv")

        val_ds = MILVideoDataset(
            json_path=os.path.join(args.annotations_path, f"split_{split}", f"mil_val_{args.dataset_name}.json"),
            num_frames=32,
            transform=val_trans,
            search_dir_paths=[args.videos_path],
        )

        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

        run_name = f"split_{split}"
        print(f"\n>>> Evaluating Run: {run_name}")

        trans_model = DinoSelfAttention(
            full_checkpoint_path=args.trans_paths[split - 1],
            num_classes=5,
        ).to(device)

        trans_results = run_inference_transformer_all(trans_model, val_loader, device, num_classes=5)

        df_trans = pd.DataFrame.from_dict(trans_results, orient='index')
        df_trans.index.name = 'video'

        prob_cols = [f"prob_{c}" for c in range(5)]
        df_trans = df_trans[['label', 'grade', 'pred_grade'] + prob_cols]
        df_trans.to_csv(csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--trans_paths', nargs='+', required=True,
                        help='Paths to the 5 per-split 5-class DinoSelfAttention checkpoints (best_sa_model.pth)')

    parser.add_argument('--dataset_name', type=str, required=True, default="paxos2020")
    parser.add_argument('--annotations_path', type=str, default="data/stratified_splits")
    parser.add_argument('--videos_path', type=str, default="data/ensemble_results_paxos2020/cleaned_videos")
    parser.add_argument('--output_dir', type=str, default="multi_eval")
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--complex_augs', action='store_true')

    args = parser.parse_args()
    main(args)