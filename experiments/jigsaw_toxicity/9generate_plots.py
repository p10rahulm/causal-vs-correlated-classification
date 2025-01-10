#!/usr/bin/env python3
"""
Generates many plots from:
  1) IMDB results (test set) from
     outputs/imdb_sentiment_combined/results_all_20250107_122057.csv
  2) Three synthetic datasets (OOD) from
    outputs/ood_sentiment_comprehensive_test/results_20250106_150841.csv

We rename experiment_type in the second file so that it matches:
    naive_baseline -> naive_baseline
    causal_phrases -> causal_subset
    regularized -> causal_mediation

We do not combine all datasets. Instead, we:
 - Keep IMDB separate
 - Keep OOD separate
 - For OOD, we never average across its three distinct sub-datasets.
"""

import os
from pathlib import Path
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Set CUDA DEVICE
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))
#!/usr/bin/env python3



# A dictionary to rename model strings
MODEL_NAME_MAP = {
    "albert": "albert",
    "bert": "bert",
    "distilbert": "distilbert",
    "electra_small_discriminator": "electra small discriminator",
    "modern_bert": "modern bert",
    "deberta": "deberta",
    "roberta": "roberta"
}

def rename_model(orig_name: str) -> str:
    """Return a spaced name for the given model key."""
    return MODEL_NAME_MAP.get(orig_name, orig_name)

def short_exptype(exptype: str, lam: float = None) -> str:
    """
    Convert experiment_type + lambda_reg to a short label:
      naive_baseline => NB
      causal_subset  => CS
      causal_mediation => "λ0.00X"
    """
    if exptype == "naive_baseline":
        return "NB"
    elif exptype == "causal_subset":
        return "CS"
    elif exptype == "causal_mediation":
        # If we have a lambda, produce "λ0.01" etc.
        # We'll handle that outside in the special bar chart.
        return "λ"
    return exptype

def main():
    # -----------------------------------
    # 1) Read IMDB data
    # -----------------------------------
    imdb_path = "outputs/imdb_sentiment_combined/results_all_20250107_122057.csv"
    df_imdb = pd.read_csv(imdb_path)
    df_imdb['dataset'] = 'IMDB'

    # Convert numeric
    numeric_cols = [
        "train_loss","test_loss","test_accuracy","test_precision",
        "test_recall","test_f1","lambda_reg","epochs","original_epochs"
    ]
    for col in numeric_cols:
        if col in df_imdb.columns:
            df_imdb[col] = pd.to_numeric(df_imdb[col], errors='coerce')

    # Rename model
    if "model" in df_imdb.columns:
        df_imdb["model"] = df_imdb["model"].apply(rename_model)

    # -----------------------------------
    # 2) Read OOD data
    # -----------------------------------
    ood_path = "outputs/ood_sentiment_comprehensive_test/results_20250106_150841.csv"
    df_ood = pd.read_csv(ood_path)

    # map exptype
    map_exptype = {
        "naive_baseline": "naive_baseline",
        "causal_phrases": "causal_subset",
        "regularized":    "causal_mediation"
    }
    if "experiment_type" in df_ood.columns:
        df_ood['experiment_type'] = df_ood['experiment_type'].replace(map_exptype)
    elif "model_type" in df_ood.columns:
        df_ood.rename(columns={"model_type":"experiment_type"}, inplace=True)
        df_ood['experiment_type'] = df_ood['experiment_type'].replace(map_exptype)

    # Convert numeric
    for col in numeric_cols:
        if col in df_ood.columns:
            df_ood[col] = pd.to_numeric(df_ood[col], errors='coerce')

    # rename model
    if "model" in df_ood.columns:
        df_ood["model"] = df_ood["model"].apply(rename_model)

    # If 'dataset' col is missing or NaN, fill with 'Unknown'
    if "dataset" not in df_ood.columns:
        df_ood["dataset"] = "Unknown"
    df_ood["dataset"] = df_ood["dataset"].fillna("Unknown")

    # -----------------------------------
    # 3) Make directories
    # -----------------------------------
    outdir_imdb = "plots/imdb"
    outdir_ood = "plots/ood"
    os.makedirs(outdir_imdb, exist_ok=True)
    os.makedirs(outdir_ood, exist_ok=True)

    def save_fig(filepath):
        """Saves as PDF and PNG, then closes."""
        plt.savefig(filepath + ".pdf", bbox_inches='tight')
        plt.savefig(filepath + ".png", bbox_inches='tight')
        plt.close()

    # ==================================================================
    # A) IMDB
    # ==================================================================

    # A0) Precision–Recall scatter per model
    df_imdb_pr = df_imdb.dropna(subset=["test_precision","test_recall"])
    if not df_imdb_pr.empty:
        for model_name, df_m in df_imdb_pr.groupby("model"):
            plt.figure(figsize=(6,5))
            sns.scatterplot(
                data=df_m,
                x="test_precision", y="test_recall",
                hue="experiment_type", style="epochs",
                s=100
            )
            plt.title(f"IMDB Precision vs Recall ({model_name})")
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.grid(True)
            outpath = os.path.join(outdir_imdb, f"imdb_precision_recall_{model_name}")
            save_fig(outpath)

    # A1) Grouped Bar Plot: test_accuracy vs epochs => horizontal axis is epochs,
    #    color is experiment_type => (5,10) * 3 => 6 bars
    df_imdb_ep = df_imdb.dropna(subset=["epochs","test_accuracy"])
    for model_name, df_m in df_imdb_ep.groupby("model"):
        plt.figure(figsize=(7,5))
        # We'll rename 'lambda_reg' axis -> 'λ Regularizer' only when needed (e.g. in line charts).
        sns.barplot(
            data=df_m, x="epochs", y="test_accuracy", hue="experiment_type",
            dodge=True
        )
        plt.title(f"IMDB: Test Accuracy by Epochs ({model_name})")
        plt.ylim(0,1)
        plt.grid(True, axis='y')
        outpath = os.path.join(outdir_imdb, f"imdb_testacc_epochs_{model_name}")
        save_fig(outpath)

    # A2) Four separate box plots for test_accuracy, test_precision, test_recall, test_f1
    #     annotated with peak value
    metrics_to_plot = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
    for metric in metrics_to_plot:
        df_plot = df_imdb.dropna(subset=[metric])
        if df_plot.empty:
            continue

        plt.figure(figsize=(8,6))
        ax = sns.boxplot(
            data=df_plot, x="model", y=metric,
            hue="experiment_type"
        )
        plt.title(f"IMDB: Box Plot of {metric} by Model & Experiment Type")
        plt.ylim(0,1)
        plt.grid(True, axis='y')
        plt.tight_layout()

        # Annotate max values:
        group_peaks = df_plot.groupby(["model","experiment_type"])[metric].max().to_dict()
        # Each box is at index i * (#exptypes) + j. 
        # We can match them to ax.artists
        boxes = ax.artists
        # We'll collect the unique levels in the same order the boxplot uses:
        model_order = df_plot["model"].unique()
        exp_order   = df_plot["experiment_type"].unique()
        num_per_x   = len(exp_order)

        for i, mod_ in enumerate(model_order):
            for j, exp_ in enumerate(exp_order):
                box_idx = i*num_per_x + j
                if box_idx < len(boxes):
                    peak_val = group_peaks.get((mod_, exp_), None)
                    if peak_val is not None:
                        box = boxes[box_idx]
                        x_center = box.get_x() + box.get_width()/2
                        plt.text(
                            x_center, peak_val+0.01, f"{peak_val:.3f}",
                            ha='center', va='bottom', fontsize=9
                        )

        outpath = os.path.join(outdir_imdb, f"imdb_box_{metric}_model_exp")
        save_fig(outpath)

    # A3) Horizontal Bar Chart for "NB" vs "CS" vs "λ..."
    df_imdb_special = df_imdb.dropna(subset=["test_accuracy"])

    for model_name, df_m in df_imdb_special.groupby("model"):
        # Gather best test_accuracy for NB and CS:
        # (We do not differentiate by epochs here, we just pick the best.)
        # Then gather all distinct lambdas for causal_mediation.
        baseline_nb = df_m[df_m["experiment_type"]=="naive_baseline"]
        best_nb = baseline_nb["test_accuracy"].max() if not baseline_nb.empty else None

        baseline_cs = df_m[df_m["experiment_type"]=="causal_subset"]
        best_cs = baseline_cs["test_accuracy"].max() if not baseline_cs.empty else None

        # gather causal_mediation
        cmed = df_m[df_m["experiment_type"]=="causal_mediation"].dropna(subset=["lambda_reg"])
        # for each lambda, best test_accuracy
        data_list = []
        if best_nb is not None:
            data_list.append(("NB", best_nb))
        if best_cs is not None:
            data_list.append(("CS", best_cs))

        # group by lambda_reg, pick best
        for lam, df_lam in cmed.groupby("lambda_reg"):
            best_val = df_lam["test_accuracy"].max()
            # label it "λ0.XXX"
            # round lam to e.g. 3 decimal places
            data_list.append((f"λ{lam:.3f}", best_val))

        # create DF for horizontal bar plot
        df_bars = pd.DataFrame(data_list, columns=["label","accuracy"])
        # sort so that NB, CS are first, then λ ascending
        def sort_key(lbl):
            # NB => 0, CS => 1, λ0. => parse float => 2+
            if lbl == "NB": return (0,0)
            if lbl == "CS": return (0,1)
            if lbl.startswith("λ"):
                # parse numeric from substring
                num = float(lbl[1:])
                return (1, num)
            return (2, 9999)
        df_bars["sort"] = df_bars["label"].apply(sort_key)
        df_bars = df_bars.sort_values("sort", ascending=True).drop(columns="sort").reset_index(drop=True)

        plt.figure(figsize=(9,5))
        ax = sns.barplot(
            data=df_bars, y="label", x="accuracy",
            color="skyblue", orient="h"
        )
        plt.title(f"IMDB Test Dataset Comparisons: {model_name}")
        plt.xlim(0,1)
        plt.grid(True, axis='x')
        plt.xlabel("Test Accuracy")

        # The user wants a horizontal line at whichever is higher among NB or CS
        nb_val = df_bars.loc[df_bars["label"]=="NB","accuracy"]
        cs_val = df_bars.loc[df_bars["label"]=="CS","accuracy"]
        if len(nb_val)>0 or len(cs_val)>0:
            # pick whichever is higher
            val_nb = nb_val.values[0] if len(nb_val)>0 else 0
            val_cs = cs_val.values[0] if len(cs_val)>0 else 0
            high_val = max(val_nb, val_cs)
            # Draw vertical line at x=high_val
            plt.axvline(x=high_val, color="red", linewidth=2)

        # annotate bar values
        for i, row in df_bars.iterrows():
            acc_val = row["accuracy"]
            y_coord = i
            ax.text(
                acc_val+0.01, y_coord, f"{acc_val:.3f}",
                va="center"
            )

        outpath = os.path.join(outdir_imdb, f"imdb_special_bar_{model_name}")
        save_fig(outpath)

    # ==================================================================
    # B) OOD
    # ==================================================================
    # We do not average across OOD datasets. We do them separately,
    # plus we create a combined scatter precision–recall across all OOD.

    # 1) per dataset
    for ds_name, df_ds in df_ood.groupby("dataset"):
        ds_dir = os.path.join(outdir_ood, ds_name.replace(" ","_").replace("/","_"))
        os.makedirs(ds_dir, exist_ok=True)

        # (a) lineplot test_accuracy vs epochs
        if "epochs" in df_ds.columns and not df_ds["epochs"].isna().all():
            for mod_, df_m in df_ds.groupby("model"):
                plt.figure(figsize=(7,5))
                sns.lineplot(
                    data=df_m, x="epochs", y="test_accuracy",
                    hue="experiment_type", marker="o"
                )
                plt.title(f"{ds_name}: Test Accuracy vs. epochs ({mod_})")
                plt.ylim(0,1)
                plt.grid(True)
                outpath = os.path.join(ds_dir, f"{mod_}_testacc_vs_epochs")
                save_fig(outpath)

        # (b) lineplot test_accuracy vs lambda_reg for causal_mediation
        df_ds_reg = df_ds[df_ds["experiment_type"]=="causal_mediation"].dropna(subset=["lambda_reg"])
        if not df_ds_reg.empty:
            for mod_, df_m in df_ds_reg.groupby("model"):
                plt.figure(figsize=(7,5))
                sns.lineplot(
                    data=df_m, x="lambda_reg", y="test_accuracy",
                    marker="o"
                )
                plt.title(f"{ds_name} {mod_} (causal_mediation): Test Accuracy vs λ Regularizer")
                plt.xlabel("λ Regularizer")
                plt.ylim(0,1)
                plt.grid(True)
                outpath = os.path.join(ds_dir, f"{mod_}_reg_testacc_vs_lambda")
                save_fig(outpath)

        # (c) boxplot test_f1
        df_f1 = df_ds.dropna(subset=["test_f1"])
        if not df_f1.empty:
            plt.figure(figsize=(9,6))
            sns.boxplot(data=df_f1, x="model", y="test_f1", hue="experiment_type")
            plt.title(f"{ds_name}: Test F1 by Model & Experiment Type")
            plt.ylim(0,1)
            plt.grid(True, axis='y')
            plt.tight_layout()
            outpath = os.path.join(ds_dir, "box_testf1_model_exp")
            save_fig(outpath)

        # (d) scatter: precision vs recall (**each** model separately)
        df_pr = df_ds.dropna(subset=["test_precision","test_recall"])
        if not df_pr.empty:
            # one scatter per model
            for mod_, df_m in df_pr.groupby("model"):
                plt.figure(figsize=(6,5))
                sns.scatterplot(
                    data=df_m,
                    x="test_precision", y="test_recall",
                    hue="experiment_type", s=100
                )
                plt.title(f"{ds_name} Precision vs Recall ({mod_})")
                plt.xlim(0,1)
                plt.ylim(0,1)
                plt.grid(True)
                outpath = os.path.join(ds_dir, f"{mod_}_scatter_precision_recall")
                save_fig(outpath)

        # (e) correlation heatmap if columns exist
        corr_cols_req = [c for c in ["train_loss","test_loss","test_accuracy",
                                     "test_precision","test_recall","test_f1"]
                         if c in df_ds.columns]
        df_ds_corr = df_ds.dropna(subset=corr_cols_req)
        if len(df_ds_corr)>1 and len(corr_cols_req)>1:
            plt.figure(figsize=(7,6))
            cmat = df_ds_corr[corr_cols_req].corr()
            sns.heatmap(cmat, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
            plt.title(f"{ds_name}: Correlation Heatmap of Metrics")
            outpath = os.path.join(ds_dir, "corr_heatmap_metrics")
            save_fig(outpath)

    # 2) **Scatter precision–recall across all OOD** for each model, color-coded by dataset
    #    => We combine all OOD data, group by model.
    df_ood_pr = df_ood.dropna(subset=["test_precision","test_recall"])
    if not df_ood_pr.empty:
        for mod_, df_m in df_ood_pr.groupby("model"):
            plt.figure(figsize=(6,5))
            sns.scatterplot(
                data=df_m,
                x="test_precision", y="test_recall",
                hue="dataset", style="experiment_type",
                s=100
            )
            plt.title(f"All OOD: Precision vs Recall ({mod_})")
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.grid(True)
            outpath = os.path.join(outdir_ood, f"allOOD_scatter_precision_recall_{mod_}")
            save_fig(outpath)

    print("All done!")
    print(f"IMDB plots => {outdir_imdb}")
    print(f"OOD plots  => {outdir_ood}")

if __name__ == "__main__":
    main()
