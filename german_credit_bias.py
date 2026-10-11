"""Bias detection and mitigation on the German credit dataset with AI Fairness 360.

The script treats age as the protected attribute (applicants 25 and older are
the privileged group, following IBM's AIF360 credit scoring tutorial). It
measures bias in the data, applies the Reweighing pre-processing algorithm,
and compares a logistic regression credit model trained on the original data
with the same model trained on the reweighed data.

The unprivileged group is small (about 15% of 1,000 applicants), so a single
train/test split gives noisy fairness numbers. Model results are therefore
averaged over 20 random 70/30 splits and reported as mean +/- standard deviation.

Data:
    UCI Statlog (German Credit Data), german.data, kept in data/. If the file
    is missing, the script downloads it from the UCI repository.

Usage:
    python german_credit_bias.py

Outputs:
    results/fairness_comparison.png   chart of original vs mitigated metrics
    results/results.txt               the same comparison as plain text
"""
import logging
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # render to file without needing a display
import matplotlib.pyplot as plt  # noqa: E402

# aif360 logs a warning for every optional extra (tensorflow, fairlearn)
# that is not installed; none are used here, so hide them during import
logging.disable(logging.WARNING)
import aif360  # noqa: E402
from aif360.datasets import GermanDataset  # noqa: E402
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric  # noqa: E402
from aif360.algorithms.preprocessing import Reweighing  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

logging.disable(logging.NOTSET)

# seeds 0..19 give 20 reproducible random train/test splits
N_SPLITS = 20
TRAIN_FRACTION = 0.7

# group definitions for age: 1 marks the privileged group (age >= 25)
PRIVILEGED = [{"age": 1}]
UNPRIVILEGED = [{"age": 0}]

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
)
PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_DATA = PROJECT_DIR / "data" / "german.data"
RESULTS_DIR = PROJECT_DIR / "results"

# chart colors: validated colorblind-safe categorical pair
COLOR_ORIGINAL = "#2a78d6"
COLOR_MITIGATED = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
LABEL_BOX = {"boxstyle": "round,pad=0.2", "facecolor": "#fcfcfb", "edgecolor": "none"}


def ensure_raw_data():
    """Put german.data where AIF360 expects it.

    AIF360 does not ship the UCI file, and GermanDataset only reads it from
    aif360/data/raw/german/ inside the installed package. The project keeps a
    copy in data/ and copies it into place, downloading it from UCI first if
    the project copy is missing.
    """
    if not LOCAL_DATA.exists():
        print(f"Downloading German credit data from UCI to {LOCAL_DATA} ...")
        LOCAL_DATA.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(UCI_URL, LOCAL_DATA)

    raw_dir = Path(aif360.__file__).resolve().parent / "data" / "raw" / "german"
    target = raw_dir / "german.data"
    if not target.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LOCAL_DATA, target)


def load_dataset():
    """Load the German credit data with age as the only protected attribute."""
    return GermanDataset(
        protected_attribute_names=["age"],
        # the tutorial's boundary: 25 and older is privileged
        privileged_classes=[lambda age: age >= 25],
        # sex is derived from personal_status, so both are dropped to keep
        # this analysis focused on age
        features_to_drop=["personal_status", "sex"],
    )


def reweigh(dataset):
    """Apply Reweighing, which assigns each (group, label) pair a weight
    so that group membership and the favorable label become independent."""
    rw = Reweighing(unprivileged_groups=UNPRIVILEGED, privileged_groups=PRIVILEGED)
    return rw.fit_transform(dataset)


def dataset_metrics(dataset):
    """Return group fairness metrics measured on a dataset's labels."""
    metric = BinaryLabelDatasetMetric(
        dataset, unprivileged_groups=UNPRIVILEGED, privileged_groups=PRIVILEGED
    )
    return {
        "base_rate_priv": metric.base_rate(privileged=True),
        "base_rate_unpriv": metric.base_rate(privileged=False),
        "spd": metric.mean_difference(),
        "di": metric.disparate_impact(),
    }


def train_and_predict(train, test, seed):
    """Fit a logistic regression credit model and predict the test set.

    Instance weights are passed to the model, so a reweighed training set
    changes what the model learns while an unweighted one trains normally.
    """
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train.features)

    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(x_train, train.labels.ravel(), sample_weight=train.instance_weights)

    predictions = test.copy(deepcopy=True)
    predictions.labels = model.predict(scaler.transform(test.features)).reshape(-1, 1)
    return predictions


def model_metrics(test, predictions):
    """Return accuracy and fairness metrics for a model's test predictions."""
    metric = ClassificationMetric(
        test, predictions, unprivileged_groups=UNPRIVILEGED, privileged_groups=PRIVILEGED
    )
    return {
        "accuracy": metric.accuracy(),
        "balanced_accuracy": 0.5 * (metric.true_positive_rate() + metric.true_negative_rate()),
        "spd": metric.statistical_parity_difference(),
        "di": metric.disparate_impact(),
        "eod": metric.equal_opportunity_difference(),
        "aod": metric.average_odds_difference(),
    }


def evaluate_splits(dataset):
    """Train original and reweighed models on each split and collect metrics.

    Reweighing is fit on the training portion only, so the test set never
    influences the mitigation.
    """
    original, mitigated = [], []
    for seed in range(N_SPLITS):
        train, test = dataset.split([TRAIN_FRACTION], shuffle=True, seed=seed)
        original.append(model_metrics(test, train_and_predict(train, test, seed)))
        mitigated.append(model_metrics(test, train_and_predict(reweigh(train), test, seed)))
    return summarize(original), summarize(mitigated)


def summarize(runs):
    """Turn a list of per-split metric dicts into {metric: (mean, std)}."""
    return {
        key: (float(np.mean([r[key] for r in runs])), float(np.std([r[key] for r in runs])))
        for key in runs[0]
    }


def format_report(counts, data_orig, data_rw, model_orig, model_rw):
    """Build the side-by-side text comparison printed to the terminal."""
    width = 40

    def single(label, orig, rw, fair):
        return f"  {label:<{width - 2}}{orig:>16.4f}{rw:>18.4f}{fair:>8}"

    def paired(label, key, fair):
        o_mean, o_sd = model_orig[key]
        m_mean, m_sd = model_rw[key]
        return (f"  {label:<{width - 2}}{o_mean:>8.4f} ± {o_sd:.3f}"
                f"{m_mean:>10.4f} ± {m_sd:.3f}{fair:>8}")

    lines = [
        "AI Fairness 360: German credit, protected attribute = age",
        f"Applicants: {counts['total']} total, {counts['priv']} age >= 25 (privileged), "
        f"{counts['unpriv']} age < 25 (unprivileged)",
        "Mitigation: Reweighing (Kamiran & Calders), pre-processing",
        "",
        f"{'Metric':<{width}}{'Original':>16}{'Mitigated':>18}{'Fair':>8}",
        "-" * (width + 42),
        "DATASET (all 1,000 applicants)",
        single("Favorable rate, age >= 25", data_orig["base_rate_priv"], data_rw["base_rate_priv"], "-"),
        single("Favorable rate, age < 25", data_orig["base_rate_unpriv"], data_rw["base_rate_unpriv"], "-"),
        single("Statistical parity difference", data_orig["spd"], data_rw["spd"], "0.00"),
        single("Disparate impact", data_orig["di"], data_rw["di"], "1.00"),
        "",
        f"CREDIT MODEL (logistic regression, mean ± SD over {N_SPLITS} test splits)",
        paired("Accuracy", "accuracy", "higher"),
        paired("Balanced accuracy", "balanced_accuracy", "higher"),
        paired("Statistical parity difference", "spd", "0.00"),
        paired("Disparate impact", "di", "1.00"),
        paired("Equal opportunity difference", "eod", "0.00"),
        paired("Average odds difference", "aod", "0.00"),
    ]
    return "\n".join(lines)


def save_chart(data_orig, data_rw, model_orig, model_rw, path):
    """Save a two-panel chart comparing original and mitigated fairness."""
    fig, (ax_diff, ax_di) = plt.subplots(
        1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [2.3, 1]}
    )
    bar_w = 0.36
    err_style = {"ecolor": INK_SECONDARY, "capsize": 3, "elinewidth": 1}

    # left panel: difference metrics, where 0 means parity
    labels = [
        "Data: statistical\nparity difference",
        "Model: statistical\nparity difference",
        "Model: equal\nopportunity difference",
        "Model: average\nodds difference",
    ]
    orig_vals = [data_orig["spd"]] + [model_orig[k][0] for k in ("spd", "eod", "aod")]
    orig_err = [np.nan] + [model_orig[k][1] for k in ("spd", "eod", "aod")]
    rw_vals = [data_rw["spd"]] + [model_rw[k][0] for k in ("spd", "eod", "aod")]
    rw_err = [np.nan] + [model_rw[k][1] for k in ("spd", "eod", "aod")]
    x = np.arange(len(labels))
    bars_o = ax_diff.bar(x - bar_w / 2 - 0.01, orig_vals, bar_w, yerr=orig_err,
                         color=COLOR_ORIGINAL, error_kw=err_style, label="Original")
    bars_m = ax_diff.bar(x + bar_w / 2 + 0.01, rw_vals, bar_w, yerr=rw_err,
                         color=COLOR_MITIGATED, error_kw=err_style, label="Mitigated (Reweighing)")
    ax_diff.axhline(0, color=INK_SECONDARY, linewidth=1)
    ax_diff.set_xticks(x, labels, fontsize=9)
    ax_diff.set_title("Difference metrics (fair value = 0)", fontsize=11, loc="left")
    # value labels sit mid-bar on a small surface-colored box so the error
    # bar line passing through them stays readable
    for bars in (bars_o, bars_m):
        for bar in bars:
            value = bar.get_height()
            y = value / 2 if abs(value) > 0.03 else -0.018
            ax_diff.text(bar.get_x() + bar.get_width() / 2, y, f"{value:.3f}", ha="center",
                         va="center", fontsize=8, color=INK, bbox=LABEL_BOX)

    # right panel: disparate impact, where 1 means parity and 0.8 marks the
    # four-fifths rule from us employment guidelines
    di_orig = [data_orig["di"], model_orig["di"][0]]
    di_rw = [data_rw["di"], model_rw["di"][0]]
    x2 = np.arange(2)
    bars_o2 = ax_di.bar(x2 - bar_w / 2 - 0.01, di_orig, bar_w, yerr=[np.nan, model_orig["di"][1]],
                        color=COLOR_ORIGINAL, error_kw=err_style)
    bars_m2 = ax_di.bar(x2 + bar_w / 2 + 0.01, di_rw, bar_w, yerr=[np.nan, model_rw["di"][1]],
                        color=COLOR_MITIGATED, error_kw=err_style)
    ax_di.axhline(1.0, color=INK_SECONDARY, linewidth=1)
    ax_di.axhline(0.8, color=INK_SECONDARY, linewidth=1, linestyle="--")
    ax_di.text(1.62, 0.815, "0.8 rule", va="bottom", ha="right", fontsize=8, color=INK_SECONDARY)
    ax_di.set_xticks(x2, ["Data", "Model"], fontsize=9)
    ax_di.set_ylim(0, 1.3)
    ax_di.set_title("Disparate impact (fair value = 1)", fontsize=11, loc="left")
    for bars in (bars_o2, bars_m2):
        for bar in bars:
            value = bar.get_height()
            ax_di.text(bar.get_x() + bar.get_width() / 2, value / 2, f"{value:.3f}", ha="center",
                       va="center", fontsize=8, color=INK, bbox=LABEL_BOX)

    for ax in (ax_diff, ax_di):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)

    fig.suptitle("German credit, protected attribute age: original vs. mitigated",
                 fontsize=12, x=0.01, ha="left", y=0.99)
    fig.legend(loc="upper right", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.99, 1.0))
    fig.text(0.01, 0.01, f"Model bars show the mean of {N_SPLITS} random 70/30 splits; "
             "error bars show one standard deviation.", fontsize=8, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    ensure_raw_data()
    dataset = load_dataset()

    # dataset-level bias before and after reweighing all 1,000 records
    data_orig = dataset_metrics(dataset)
    data_rw = dataset_metrics(reweigh(dataset))

    # model-level bias averaged over repeated random splits
    model_orig, model_rw = evaluate_splits(dataset)

    age_col = dataset.protected_attribute_names.index("age")
    priv = int((dataset.protected_attributes[:, age_col] == 1).sum())
    counts = {"total": len(dataset.labels), "priv": priv, "unpriv": len(dataset.labels) - priv}

    report = format_report(counts, data_orig, data_rw, model_orig, model_rw)
    print(report)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "results.txt").write_text(report + "\n", encoding="utf-8")
    save_chart(data_orig, data_rw, model_orig, model_rw, RESULTS_DIR / "fairness_comparison.png")
    print("\nSaved results/fairness_comparison.png and results/results.txt")


if __name__ == "__main__":
    main()
