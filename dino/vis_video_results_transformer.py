import torch
import pandas as pd
import os
from tqdm import tqdm
import argparse
from pathlib import Path

from data.dataset import MILVideoDataset
from dino.utils import make_val_transform_dino
from dino.train_dino_mil import DinoMIL
from dino.train_dino_transformer import DinoSelfAttention

from dino.train_dino_classifier import DinoClassifier
from dino.vis_video_results import run_inference_transformer


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

        # trans_model_old = DinoSelfAttention(checkpoint_path=args.trans_paths_old[split - 1]).to(device)
        # trans_results_old = run_inference_transformer(trans_model_old, val_loader, device)

        trans_model_pos_enc = DinoSelfAttention(full_checkpoint_path=args.trans_paths_pos_enc[split - 1],
                                                use_pos_embedding=True).to(device)
        trans_results_pos_enc = run_inference_transformer(trans_model_pos_enc, val_loader, device)

        trans_model_deeper = DinoSelfAttention(full_checkpoint_path=args.trans_paths_deeper[split - 1],
                                               num_blocks=4).to(device)
        trans_results_deeper = run_inference_transformer(trans_model_deeper, val_loader, device)

        # df_trans_old = pd.DataFrame.from_dict(trans_results_old, orient='index')
        # df_trans_old = df_trans_old.rename(columns={'prob': f"Trans_Old_{run_name}"})

        df_trans_pos_enc = pd.DataFrame.from_dict(trans_results_pos_enc, orient='index')
        df_trans_pos_enc = df_trans_pos_enc.rename(columns={'prob': f"Trans_Pos_Enc_{run_name}"})

        df_trans_deeper = pd.DataFrame.from_dict(trans_results_deeper, orient='index')
        df_trans_deeper = df_trans_deeper.rename(columns={'prob': f"Trans_Deeper_{run_name}"})

        # df_trans_old = df_trans_old.drop(columns=['label']).drop(columns=['grade'])
        df_trans_pos_enc = df_trans_pos_enc.drop(columns=['label']).drop(columns=['grade'])

        # Join both frames on the video name index
        # master_df = df_trans_deeper.join(df_trans_pos_enc).join(df_trans_old)
        master_df = df_trans_deeper.join(df_trans_pos_enc)
        master_df.index.name = 'video'

        # Reorder columns to look clean: [label, MIL_split_X, Clf_split_X, Trans_split_X]
        # master_df = master_df[['label', f"Trans_Old_{run_name}", f"Trans_Pos_Enc_{run_name}", f"Trans_Deeper_{run_name}", "grade"]]
        master_df = master_df[['label', f"Trans_Pos_Enc_{run_name}", f"Trans_Deeper_{run_name}", "grade"]]
        master_df.to_csv(csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--trans_paths_old', nargs='+', required=True)
    parser.add_argument('--trans_paths_pos_enc', nargs='+', required=True)
    parser.add_argument('--trans_paths_deeper', nargs='+', required=True)

    parser.add_argument('--dataset_name', type=str, required=True, default="paxos2020")
    parser.add_argument('--annotations_path', type=str, default="data/stratified_splits")
    parser.add_argument('--videos_path', type=str, default="data/ensemble_results_paxos2020/cleaned_videos")
    parser.add_argument('--output_dir', type=str, default="multi_eval")
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--complex_augs', action='store_true')

    args = parser.parse_args()
    main(args)
