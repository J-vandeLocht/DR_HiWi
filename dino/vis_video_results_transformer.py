import torch
import pandas as pd
import os
from tqdm import tqdm
import argparse
from pathlib import Path

from data.dataset import MILVideoDatasetRope
from dino.utils import make_val_transform_dino
from dino.train_dino_mil import DinoMIL
from dino.train_dino_transformer import DinoSelfAttention
from dino.train_dino_transformer_rope import DinoSelfAttention as DinoSelfAttentionROPE

from dino.train_dino_classifier import DinoClassifier
from dino.vis_video_results import run_inference_transformer, run_inference_transformer_rope


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs)

    for split in [1, 2, 3, 4, 5]:
        csv_path = os.path.join(args.output_dir, f"predictions_split_{split}_{args.dataset_name}.csv")

        val_ds = MILVideoDatasetRope(
            json_path=os.path.join(args.annotations_path, f"split_{split}", f"mil_val_{args.dataset_name}.json"),
            num_frames=32,
            transform=val_trans,
            search_dir_paths=[args.videos_path],
        )

        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

        run_name = f"split_{split}"
        print(f"\n>>> Evaluating Run: {run_name}")

        trans_model = DinoSelfAttention(full_checkpoint_path=args.trans_paths[split - 1]).to(device)
        trans_results = run_inference_transformer(trans_model, val_loader, device)

        trans_model_rope_small = DinoSelfAttentionROPE(full_checkpoint_path=args.trans_paths_rope_small[split - 1],
                                                       use_rope=True,
                                                       rope_base=10_000,
                                                       rope_position_scale=31,
                                                       ).to(device)
        trans_results_rope_small = run_inference_transformer_rope(trans_model_rope_small, val_loader, device)

        trans_model_rope_large = DinoSelfAttentionROPE(full_checkpoint_path=args.trans_paths_rope_large[split - 1],
                                                       use_rope=True,
                                                       rope_base=100,
                                                       rope_position_scale=300,
                                                       ).to(device)
        trans_results_rope_large = run_inference_transformer_rope(trans_model_rope_large, val_loader, device)

        df_trans = pd.DataFrame.from_dict(trans_results, orient='index')
        df_trans = df_trans.rename(columns={'prob': f"Trans_{run_name}"})

        df_trans_rope_small = pd.DataFrame.from_dict(trans_results_rope_small, orient='index')
        df_trans_rope_small = df_trans_rope_small.rename(columns={'prob': f"Trans_Rope_Small_{run_name}"})

        df_trans_rope_large = pd.DataFrame.from_dict(trans_results_rope_large, orient='index')
        df_trans_rope_large = df_trans_rope_large.rename(columns={'prob': f"Trans_Rope_Large_{run_name}"})

        df_trans = df_trans.drop(columns=['label']).drop(columns=['grade'])
        df_trans_rope_large = df_trans_rope_large.drop(columns=['label']).drop(columns=['grade'])

        # Join both frames on the video name index
        master_df = df_trans_rope_small.join(df_trans).join(df_trans_rope_large)
        master_df.index.name = 'video'

        # Reorder columns to look clean: [label, MIL_split_X, Clf_split_X, Trans_split_X]
        master_df = master_df[['label', "grade", f"Trans_{run_name}", f"Trans_Rope_Small_{run_name}", f"Trans_Rope_Large_{run_name}"]]
        master_df.to_csv(csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--trans_paths', nargs='+', required=True)
    parser.add_argument('--trans_paths_rope_small', nargs='+', required=True)
    parser.add_argument('--trans_paths_rope_large', nargs='+', required=True)

    parser.add_argument('--dataset_name', type=str, required=True, default="paxos2020")
    parser.add_argument('--annotations_path', type=str, default="data/stratified_splits")
    parser.add_argument('--videos_path', type=str, default="data/ensemble_results_paxos2020/cleaned_videos")
    parser.add_argument('--output_dir', type=str, default="multi_eval")
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--complex_augs', action='store_true')

    args = parser.parse_args()
    main(args)
