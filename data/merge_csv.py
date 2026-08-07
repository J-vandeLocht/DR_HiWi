import pandas as pd


def main():
    for split in range(1, 6):
        old_tbl = pd.read_csv(f'multi_eval_paxos2025/predictions_split_{split}_paxos2025.csv')
        new_tbl = pd.read_csv(f"multi_eval_paxos2025_transformer/predictions_split_{split}_paxos2025.csv")
        old_tbl = old_tbl.drop(columns=['label', 'grade', f'MIL_split_{split}', f'Clf_split_{split}'])
        new_tbl = new_tbl.merge(old_tbl, on='video', how='left')
        new_tbl = new_tbl[['video', 'label', 'grade', f"Trans_split_{split}", f"Trans_Pos_Enc_split_{split}", f"Trans_Deeper_split_{split}"]]
        new_tbl.to_csv(f"predictions_split_{split}_paxos2025.csv", index=False)


if __name__ == '__main__':
    main()