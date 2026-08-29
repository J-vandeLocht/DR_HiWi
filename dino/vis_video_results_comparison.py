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
from dino.vis_video_results import run_inference_classifier, run_inference_transformer


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

        clf_2020_model = DinoClassifier(freeze_backbone=True, num_classes=1).to(device)
        state_dict = torch.load(args.clf_2020_paths[split - 1], map_location=device)
        clf_2020_model.load_state_dict(state_dict)
        clf_2020_results = run_inference_classifier(clf_2020_model, val_loader, device)
        clf_2025_model = DinoClassifier(freeze_backbone=True, num_classes=1).to(device)
        state_dict = torch.load(args.clf_2025_paths[split - 1], map_location=device)
        clf_2025_model.load_state_dict(state_dict)
        clf_2025_results = run_inference_classifier(clf_2025_model, val_loader, device)
        clf_fused_model = DinoClassifier(freeze_backbone=True, num_classes=1).to(device)
        state_dict = torch.load(args.clf_fused_paths[split - 1], map_location=device)
        clf_fused_model.load_state_dict(state_dict)
        clf_fused_results = run_inference_classifier(clf_fused_model, val_loader, device)

        trans_2020_model = DinoSelfAttention(full_checkpoint_path=args.trans_2020_paths[split - 1]).to(device)
        trans_2020_results = run_inference_transformer(trans_2020_model, val_loader, device)
        trans_2025_model = DinoSelfAttention(full_checkpoint_path=args.trans_2025_paths[split - 1]).to(device)
        trans_2025_results = run_inference_transformer(trans_2025_model, val_loader, device)
        trans_fused_model = DinoSelfAttention(full_checkpoint_path=args.trans_fused_paths[split - 1]).to(device)
        trans_fused_results = run_inference_transformer(trans_fused_model, val_loader, device)

        df_clf_2020 = pd.DataFrame.from_dict(clf_2020_results, orient='index')
        df_clf_2020 = df_clf_2020.rename(columns={'prob': f"Clf_2020_{run_name}"})
        df_clf_2025 = pd.DataFrame.from_dict(clf_2025_results, orient='index')
        df_clf_2025 = df_clf_2025.rename(columns={'prob': f"Clf_2025_{run_name}"})
        df_clf_fused = pd.DataFrame.from_dict(clf_fused_results, orient='index')
        df_clf_fused = df_clf_fused.rename(columns={'prob': f"Clf_fused_{run_name}"})

        df_trans_2020 = pd.DataFrame.from_dict(trans_2020_results, orient='index')
        df_trans_2020 = df_trans_2020.rename(columns={'prob': f"Trans_2020_{run_name}"})
        df_trans_2025 = pd.DataFrame.from_dict(trans_2025_results, orient='index')
        df_trans_2025 = df_trans_2025.rename(columns={'prob': f"Trans_2025_{run_name}"})
        df_trans_fused = pd.DataFrame.from_dict(trans_fused_results, orient='index')
        df_trans_fused = df_trans_fused.rename(columns={'prob': f"Trans_fused_{run_name}"})

        df_trans_2020 = df_trans_2020.drop(columns=['label']).drop(columns=['grade'])
        df_trans_2025 = df_trans_2025.drop(columns=['label']).drop(columns=['grade'])
        df_trans_fused = df_trans_fused.drop(columns=['label']).drop(columns=['grade'])

        df_clf_2020 = df_clf_2020.drop(columns=['label']).drop(columns=['grade'])
        df_clf_2025 = df_clf_2025.drop(columns=['label']).drop(columns=['grade'])

        master_df = df_clf_fused.join(df_clf_2020).join(df_clf_2025).join(df_trans_2020).join(df_trans_2025).join(df_trans_fused)
        master_df.index.name = 'video'

        master_df = master_df[['label', 'grade',
                               f"Clf_2020_{run_name}", f"Trans_2020_{run_name}",
                               f"Clf_2025_{run_name}", f"Trans_2025_{run_name}",
                               f"Clf_fused_{run_name}", f"Trans_fused_{run_name}"]]
        master_df.to_csv(csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--clf_2020_paths', nargs='+', required=True)
    parser.add_argument('--clf_2025_paths', nargs='+', required=True)
    parser.add_argument('--clf_fused_paths', nargs='+', required=True)
    parser.add_argument('--trans_2020_paths', nargs='+', required=True)
    parser.add_argument('--trans_2025_paths', nargs='+', required=True)
    parser.add_argument('--trans_fused_paths', nargs='+', required=True)

    parser.add_argument('--dataset_name', type=str, required=True, default="paxos2020")
    parser.add_argument('--annotations_path', type=str, default="data/stratified_splits")
    parser.add_argument('--videos_path', type=str, default="data/ensemble_results_paxos2020/cleaned_videos")
    parser.add_argument('--output_dir', type=str, default="multi_eval")
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--complex_augs', action='store_true')

    args = parser.parse_args()
    main(args)
