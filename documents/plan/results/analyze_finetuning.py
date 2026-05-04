"""
analyze_finetuning.py
---------------------
Liest alle single_finetune/<dataset>/<freq?>/evaluation.csv Dateien,
berechnet MASE-Verbesserung/-Verschlechterung relativ zum Originalmodell
(finetuning_place == -1) und erzeugt 6 Plots:

    1. 3D-Scatter        - context_length x num_series x delta_pct
    2. Checkpoint-Linien - MASE-Delta pro Dataset ueber Checkpoints
    3. PCA               - (context_length, num_series) -> 2D, Farbe = mittleres delta_pct
    4. Spearman-Barplot  - Korrelation der Features mit delta_pct
    5. 2D-Scatter        - context_length x num_series, Farbe = mittleres delta_pct
    6. OLS-Regression    - Standardisierte Beta-Koeffizienten mit 95%-KI

Ausfuehrung (vom Projektroot):
    python analyze_finetuning.py [--root single_finetune] [--save]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

try:
    from finetune_tabpfn_ts.evaluation.data import Dataset, Term
    HAS_DATASET = True
except ImportError:
    warnings.warn(
        "finetune_tabpfn_ts konnte nicht importiert werden. "
        "context_length / num_series werden aus der CSV gelesen.",
        stacklevel=2,
    )
    HAS_DATASET = False


# =============================================================================
# 1. Daten laden
# =============================================================================

def get_dataset_stats(dataset_name: str) -> tuple[int, int]:
    ds = Dataset(name=dataset_name, term=Term.SHORT, to_univariate=False)
    context_length = ds._min_series_length
    num_series = sum(1 for _ in ds.gluonts_dataset)
    return context_length, num_series


def find_evaluation_csvs(root: Path) -> list[Path]:
    root = root.resolve()
    found = sorted(root.rglob("evaluation.csv"))
    print("Durchsuche: " + str(root))
    if not found:
        print("  -> Keine evaluation.csv gefunden.")
    else:
        for p in found:
            print("  OK " + str(p.relative_to(root)))
    return found


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def compute_mase_delta(df: pd.DataFrame) -> pd.DataFrame:
    mase_col = "eval_metrics/MASE[0.5]"
    original = df[df["finetuning_place"] == -1]
    if original.empty:
        raise ValueError("Kein Eintrag mit finetuning_place == -1 gefunden.")
    mase_orig = original[mase_col].values[0]
    finetuned = df[df["finetuning_place"] != -1].copy()
    finetuned["mase_original"] = mase_orig
    finetuned["mase_ft"] = finetuned[mase_col]
    finetuned["delta_pct"] = (finetuned["mase_ft"] - mase_orig) / mase_orig * 100.0
    return finetuned[["finetuning_place", "mase_original", "mase_ft", "delta_pct"]]


def build_records(root: Path) -> pd.DataFrame:
    records = []
    csvs = find_evaluation_csvs(root)
    if not csvs:
        raise FileNotFoundError("Keine evaluation.csv unter " + str(root) + " gefunden.")

    print("Gefundene evaluation.csv Dateien: " + str(len(csvs)) + "\n")

    TERM_NAMES = {"short", "medium", "long"}

    for csv_path in csvs:
        try:
            df = load_csv(csv_path)
            deltas = compute_mase_delta(df)
        except Exception as e:
            warnings.warn("Fehler beim Lesen von " + str(csv_path) + ": " + str(e))
            continue

        # Dataset-Name aus Ordnerstruktur:
        # single_finetune/<dataset>/<frequency>/<term>/evaluation.csv -> "<dataset>/<frequency>"
        # single_finetune/<dataset>/<term>/evaluation.csv             -> "<dataset>"
        parts = csv_path.relative_to(root).parts[:-1]  # ohne "evaluation.csv"
        meaningful = [p for p in parts if p.lower() not in TERM_NAMES]
        dataset_name = "/".join(meaningful)

        print("Verarbeite: " + dataset_name + "  (" + str(csv_path.relative_to(root)) + ")")

        if HAS_DATASET:
            try:
                context_length, num_series = get_dataset_stats(dataset_name)
            except Exception as e:
                warnings.warn("Dataset-Stats fuer '" + dataset_name + "' nicht ladbar: " + str(e))
                context_length = df.get("context_length", pd.Series([np.nan])).iloc[0]
                num_series = df.get("num_variates", pd.Series([np.nan])).iloc[0]
        else:
            context_length = df.get("context_length", pd.Series([np.nan])).iloc[0]
            num_series = df.get("num_variates", pd.Series([np.nan])).iloc[0]

        for _, row in deltas.iterrows():
            records.append({
                "dataset": dataset_name,
                "finetuning_place": int(row["finetuning_place"]),
                "mase_original": row["mase_original"],
                "mase_finetuned": row["mase_ft"],
                "delta_pct": row["delta_pct"],
                "context_length": context_length,
                "num_series": num_series,
            })

    return pd.DataFrame(records)


# =============================================================================
# 2. Terminal-Ausgabe
# =============================================================================

def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG  (delta_pct: negativ = Verbesserung)")
    print("=" * 70)
    summary = (
        df.groupby(["dataset", "finetuning_place"])
        .agg(
            delta_pct=("delta_pct", "first"),
            context_length=("context_length", "first"),
            num_series=("num_series", "first"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))
    print()
    print("Pro Finetuning-Platz (Mittelwert ueber alle Datasets):")
    print(df.groupby("finetuning_place")["delta_pct"].describe().to_string())
    print()
    print("-" * 70)
    print("SPEARMAN-KORRELATION mit delta_pct:")
    for feat in ["context_length", "num_series", "finetuning_place"]:
        r, p = stats.spearmanr(df[feat], df["delta_pct"])
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        print("  {:<22} r={:+.3f}  p={:.4f}  {}".format(feat, r, p, sig))
    print()


# =============================================================================
# 3. Hilfsfunktion: symmetrische Norm um 0
# =============================================================================

def sym_norm(values):
    """Gibt plt.Normalize zurueck mit -abs_max .. +abs_max, sodass 0 immer Mitte ist."""
    abs_max = max(abs(float(values.min())), abs(float(values.max())))
    if abs_max == 0:
        abs_max = 1.0
    return plt.Normalize(vmin=-abs_max, vmax=abs_max)


def dataset_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregiert delta_pct pro Dataset (Mittelwert ueber alle Checkpoints)."""
    return (
        df.groupby("dataset")
        .agg(
            context_length=("context_length", "first"),
            num_series=("num_series", "first"),
            delta_mean=("delta_pct", "mean"),
            delta_min=("delta_pct", "min"),
            delta_max=("delta_pct", "max"),
        )
        .reset_index()
    )


# =============================================================================
# 4. Die 6 Plot-Funktionen
# =============================================================================

def plot_3d(df: pd.DataFrame, ax3d):
    """① 3D-Scatter: context_length x num_series x delta_pct, Farbe pro Checkpoint"""
    places = sorted(df["finetuning_place"].unique())
    cmap = cm.viridis
    colors = {p: cmap(i / max(len(places) - 1, 1)) for i, p in enumerate(places)}

    for place in places:
        sub = df[df["finetuning_place"] == place]
        ax3d.scatter(
            sub["context_length"], sub["num_series"], sub["delta_pct"],
            c=[colors[place]], label="Ckpt " + str(place), s=55, alpha=0.85,
        )

    xl, yl = ax3d.get_xlim(), ax3d.get_ylim()
    xx, yy = np.meshgrid(xl, yl)
    ax3d.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.12, color="red")

    ax3d.set_xlabel("Context Length", fontsize=8, labelpad=8)
    ax3d.set_ylabel("Num Series", fontsize=8, labelpad=8)
    ax3d.set_zlabel("MASE delta%", fontsize=8, labelpad=8)
    ax3d.set_title("① 3D-Scatter", fontsize=10, fontweight="bold")
    ax3d.legend(fontsize=7, loc="upper left")


def plot_checkpoints(df: pd.DataFrame, ax):
    """② Linienplot: MASE-Delta pro Dataset ueber Checkpoints"""
    datasets = df["dataset"].unique()
    cmap = cm.tab20
    for i, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds].sort_values("finetuning_place")
        ax.plot(
            sub["finetuning_place"], sub["delta_pct"],
            marker="o", label=ds,
            color=cmap(i / max(len(datasets) - 1, 1)),
            alpha=0.8, linewidth=1.4,
        )
    ax.axhline(0, color="red", lw=1.2, ls="--")
    ax.set_xlabel("Finetuning Checkpoint", fontsize=9)
    ax.set_ylabel("MASE delta% vs Original", fontsize=9)
    ax.set_title("② MASE-Delta pro Checkpoint & Dataset", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_pca(df: pd.DataFrame, ax):
    """③ PCA: (context_length, num_series) -> 2D.
    Farbe = mittleres delta_pct pro Dataset (Mittelwert ueber alle Checkpoints).
    Korrekte symmetrische Norm: negativ = gruen, positiv = rot, 0 = gelb.
    """
    agg = dataset_agg(df)

    features = agg[["context_length", "num_series"]].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(features_scaled)
    explained = pca.explained_variance_ratio_ * 100

    agg = agg.copy()
    agg["pc1"] = coords[:, 0]
    agg["pc2"] = coords[:, 1]

    norm = sym_norm(agg["delta_mean"])
    sc = ax.scatter(
        agg["pc1"], agg["pc2"],
        c=agg["delta_mean"], cmap="RdYlGn_r", norm=norm,
        s=90, alpha=0.9, edgecolors="k", linewidths=0.5,
    )
    plt.colorbar(sc, ax=ax, label="Mittleres MASE delta%", shrink=0.85)

    # Loadings-Pfeile
    loadings = pca.components_.T
    scale = max(abs(coords).max() * 0.5, 1.0)
    for j, fname in enumerate(["ctx_len", "num_series"]):
        ax.annotate(
            "",
            xy=(loadings[j, 0] * scale, loadings[j, 1] * scale),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.8),
        )
        ax.text(
            loadings[j, 0] * scale * 1.12,
            loadings[j, 1] * scale * 1.12,
            fname, color="blue", fontsize=8,
        )

    # Dataset-Labels mit mittlerem delta
    for _, row in agg.iterrows():
        label = row["dataset"] + "\n(" + "{:+.1f}%".format(row["delta_mean"]) + ")"
        ax.annotate(
            label,
            (row["pc1"], row["pc2"]),
            fontsize=5, alpha=0.65, xytext=(3, 3),
            textcoords="offset points",
        )

    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("PC1 ({:.1f}% Var)".format(explained[0]), fontsize=9)
    ax.set_ylabel("PC2 ({:.1f}% Var)".format(explained[1]), fontsize=9)
    ax.set_title(
        "③ PCA (context_length, num_series -> 2D)\nFarbe = mittleres MASE delta% pro Dataset",
        fontsize=10, fontweight="bold",
    )


def plot_spearman(df: pd.DataFrame, ax):
    """④ Spearman-Korrelations-Barplot mit Signifikanz-Sternen"""
    features = ["context_length", "num_series", "finetuning_place"]
    labels   = ["Context\nLength", "Num\nSeries", "Checkpoint"]

    results = [stats.spearmanr(df[f], df["delta_pct"]) for f in features]
    rs = [r for r, _ in results]
    ps = [p for _, p in results]
    bar_colors = ["#d73027" if r > 0 else "#1a9850" for r in rs]

    bars = ax.bar(labels, rs, color=bar_colors, alpha=0.85, edgecolor="k", linewidth=0.8)

    for bar, (r, p) in zip(bars, results):
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        offset = 0.04 if r >= 0 else -0.06
        va = "bottom" if r >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            r + offset,
            sig + "\nr={:+.2f}".format(r),
            ha="center", va=va, fontsize=8,
        )

    for i, p in enumerate(ps):
        ax.text(i, -1.05, "p={:.3f}".format(p), ha="center", fontsize=7, color="gray")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(-1.15, 1.15)
    ax.set_ylabel("Spearman r mit MASE delta%", fontsize=9)
    ax.set_title(
        "④ Spearman-Korrelation\n(gruen = Verbesserung bei hoeherem Wert)",
        fontsize=10, fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3)


def plot_2d_scatter(df: pd.DataFrame, ax):
    """⑤ 2D-Scatter: context_length x num_series.
    Farbe = mittleres delta_pct ueber alle Checkpoints pro Dataset.
    Ein Punkt pro Dataset -> kein Uebereinanderstapeln.
    Korrekte symmetrische Norm: negativ = gruen, positiv = rot, 0 = gelb.
    """
    agg = dataset_agg(df)

    norm = sym_norm(agg["delta_mean"])
    cmap_used = cm.RdYlGn_r
    point_colors = [cmap_used(norm(v)) for v in agg["delta_mean"]]

    ax.scatter(
        agg["context_length"], agg["num_series"],
        c=point_colors,
        marker="o", s=120, alpha=0.9, edgecolors="k", linewidths=0.8,
    )

    sm_obj = cm.ScalarMappable(cmap=cmap_used, norm=norm)
    sm_obj.set_array([])
    plt.colorbar(sm_obj, ax=ax, label="Mittleres MASE delta%\n(negativ = Verbesserung)", shrink=0.85)

    for _, row in agg.iterrows():
        label = row["dataset"] + "\n(" + "{:+.1f}%".format(row["delta_mean"]) + ")"
        ax.annotate(
            label,
            (row["context_length"], row["num_series"]),
            fontsize=5, alpha=0.7, xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("Context Length", fontsize=9)
    ax.set_ylabel("Num Time Series", fontsize=9)
    ax.set_title(
        "⑤ 2D-Scatter (verlustfrei)\nFarbe = mittleres MASE delta% ueber alle Checkpoints",
        fontsize=10, fontweight="bold",
    )
    ax.grid(True, alpha=0.25)


def plot_regression(df: pd.DataFrame, ax):
    """⑥ OLS-Regression: standardisierte Beta-Koeffizienten mit 95%-KI"""
    X = df[["context_length", "num_series", "finetuning_place"]].copy()
    y = df["delta_pct"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_sm = sm.add_constant(X_scaled)
    model = sm.OLS(y, X_sm).fit()

    feature_names = ["Context\nLength", "Num\nSeries", "Checkpoint"]
    coefs = model.params[1:]
    conf  = model.conf_int()[1:]
    pvals = model.pvalues[1:]

    lower_err = coefs - conf[:, 0]
    upper_err = conf[:, 1] - coefs
    bar_colors = ["#d73027" if c > 0 else "#1a9850" for c in coefs]

    x_pos = np.arange(len(feature_names))
    ax.bar(
        x_pos, coefs,
        color=bar_colors, alpha=0.8, edgecolor="k", linewidth=0.8,
        yerr=[lower_err, upper_err],
        error_kw=dict(elinewidth=1.5, capsize=6, ecolor="black"),
    )

    err_max = float(max(upper_err))
    for xi, (c, p) in enumerate(zip(coefs, pvals)):
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        offset = err_max * 0.2 * float(np.sign(c)) if c != 0 else err_max * 0.2
        ax.text(
            xi, c + offset,
            sig + "\nb={:+.2f}".format(c),
            ha="center", fontsize=8,
        )

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names, fontsize=9)
    ax.set_ylabel("Standardisierter Koeffizient b\n(MASE delta%)", fontsize=9)
    ax.set_title(
        "⑥ OLS-Regression (standardisiert)\n"
        "R2={:.3f}  |  R2adj={:.3f}  |  negativ b = Verbesserung".format(
            model.rsquared, model.rsquared_adj
        ),
        fontsize=10, fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3)

    print("\n" + "=" * 70)
    print("OLS REGRESSION SUMMARY")
    print("=" * 70)
    print(model.summary())


# =============================================================================
# 5. Alle 6 Plots zusammen rendern
# =============================================================================

def render_all_plots(df: pd.DataFrame, save_path=None):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        "Finetuning-Analyse: Context Length & Num Series -> MASE-Verbesserung",
        fontsize=14, fontweight="bold", y=1.01,
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    plot_3d(df, ax1)
    plot_checkpoints(df, ax2)
    plot_pca(df, ax3)
    plot_spearman(df, ax4)
    plot_2d_scatter(df, ax5)
    plot_regression(df, ax6)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Gesamtplot gespeichert: " + str(save_path))

    plt.show()


# =============================================================================
# 6. CSV-Export
# =============================================================================

def export_csv(df: pd.DataFrame, path: Path):
    """
    Schreibt eine CSV mit allen 5 Checkpoint-Ergebnissen pro Dataset.

    Spalten:
        dataset, context_length, num_series,
        ckpt_1_delta, ckpt_2_delta, ckpt_3_delta, ckpt_4_delta, ckpt_5_delta,
        mean_delta, min_delta, max_delta,
        improved  (True wenn mean_delta < 0)
    """
    places = sorted(df["finetuning_place"].unique())

    rows = []
    for ds, grp in df.groupby("dataset"):
        grp = grp.sort_values("finetuning_place")
        row = {
            "dataset":        ds,
            "context_length": grp["context_length"].iloc[0],
            "num_series":     grp["num_series"].iloc[0],
        }
        for place in places:
            sub = grp[grp["finetuning_place"] == place]
            delta = sub["delta_pct"].values[0] if len(sub) > 0 else float("nan")
            row["ckpt_{}_delta_pct".format(place)] = round(delta, 4)

        deltas = grp["delta_pct"].values
        row["mean_delta_pct"] = round(float(deltas.mean()), 4)
        row["min_delta_pct"]  = round(float(deltas.min()),  4)
        row["max_delta_pct"]  = round(float(deltas.max()),  4)
        row["improved"]       = row["mean_delta_pct"] < 0
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    out.to_csv(path, index=False)
    print("CSV gespeichert: " + str(path))
    print(out.to_string(index=False))


# =============================================================================
# 7. Entrypoint
# =============================================================================

def main():
    SCRIPT_DIR = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Finetuning MASE Analysis - 6 Plots")
    parser.add_argument(
        "--root", type=Path, default=SCRIPT_DIR / "single_finetune",
        help="Wurzelordner der Evaluierungsergebnisse (default: <script-dir>/single_finetune)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Plot als PNG speichern -> analysis_plots.png",
    )
    args = parser.parse_args()

    df = build_records(args.root)
    if df.empty:
        print("Keine Daten geladen. Bitte Pfad und CSV-Struktur pruefen.")
        return

    df_clean = df.dropna(subset=["context_length", "num_series"])
    n_dropped = len(df) - len(df_clean)
    if n_dropped:
        warnings.warn(str(n_dropped) + " Zeilen wegen fehlender context_length/num_series verworfen.")

    print_summary(df_clean)
    export_csv(df_clean, SCRIPT_DIR / "finetuning_results.csv")
    save_path = Path("analysis_plots.png") if args.save else None
    render_all_plots(df_clean, save_path=save_path)


if __name__ == "__main__":
    main()