from pathlib import Path

import matplotlib
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


folder = Path(__file__).resolve().parent
data_dir = folder / "results"
output_dir = folder / "graphs"

buildings = ["Comms", "Library", "SSE"]
models = [
    "POOLED_anchored",
    "POOLED_shared_offset",
    "POOLED_building_offsets",
]
colors = ["#b74735", "#176b87", "#6a4c93"]


def residual_plot():
    data = pd.read_csv(data_dir / "binned_residuals.csv")
    fig, axes = plt.subplots(
        1, 3, figsize=(7, 1.9), sharey=True, layout="constrained"
    )

    for ax, building in zip(axes, buildings):
        for model, color in zip(models, colors):
            values = data[
                (data["building"] == building) & (data["spec"] == model)
            ]
            ax.plot(
                values["distance_m"],
                values["mean_residual_dB"],
                "o-",
                markersize=2.5,
                color=color,
                label=model.removeprefix("POOLED_").replace("_", " "),
            )

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(building)
        ax.set_xlabel("Distance (m)")

    axes[0].set_ylabel("Mean residual (dB)")
    axes[1].legend(
        fontsize=5.5, frameon=False, loc="upper right", handlelength=1
    )
    fig.savefig(output_dir / "residual_distance.png")
    plt.close(fig)


def variogram_plot():
    data = pd.read_csv(data_dir / "variograms.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7, 3.05), layout="constrained")

    for column, building in enumerate(buildings):
        for row, config in enumerate(["C1", "C2"]):
            ax = axes[row, column]

            for model, color in zip(models[1:], colors[1:]):
                values = data[
                    (data["building"] == building)
                    & (data["config"] == config)
                    & (data["spec"] == model)
                ]
                ax.plot(
                    values["lag_m"],
                    values["semivariance_dB2"],
                    "o-",
                    markersize=2,
                    color=color,
                    label=model.removeprefix("POOLED_").replace("_", " "),
                )

            ax.set_title(f"{building} {config}")
            ax.locator_params(axis="both", nbins=4)

            if row == 1:
                ax.set_xlabel("Separation (m)")
            if column == 0:
                ax.set_ylabel("Semivariance (dB²)")

    axes[0, 1].legend(
        fontsize=5.5, frameon=False, loc="lower right", handlelength=1
    )
    fig.savefig(output_dir / "variograms.png")
    plt.close(fig)


def qq_plot():
    data = pd.read_csv(data_dir / "residuals.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7, 1.9), layout="constrained")

    for ax, building in zip(axes, buildings):
        residuals = data.loc[
            data["building"] == building, "POOLED_shared_offset"
        ]
        (quantiles, values), (slope, intercept, _) = stats.probplot(residuals)

        ax.plot(quantiles, values, "o", markersize=2, color=colors[1])
        ax.plot(
            quantiles,
            intercept + slope * quantiles,
            color=colors[0],
            linewidth=0.8,
        )
        ax.set_title(building)
        ax.set_xlabel("Normal quantile")
        ax.locator_params(axis="both", nbins=4)

    axes[0].set_ylabel("Residual (dB)")
    fig.savefig(output_dir / "qq_plots.png")
    plt.close(fig)


def dual_slope_plot():
    data = pd.read_csv(data_dir / "spec_results_full_precision.csv")
    data = data[data["spec"].str.startswith("Comms_dual_")]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.1), layout="constrained")

    axes[0].errorbar(
        data["breakpoint_m"],
        data["offset_dB"],
        yerr=[
            data["offset_dB"] - data["offset_ci_lo"],
            data["offset_ci_hi"] - data["offset_dB"],
        ],
        fmt="o",
        markersize=3,
        capsize=3,
        color=colors[1],
    )
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_ylabel("Offset above anchor (dB)")

    slopes = [
        ("n_hat", "n_ci_lo", "n_ci_hi", -0.25, colors[1], "Below break"),
        ("n_long", "n_long_ci_lo", "n_long_ci_hi", 0.25, "#c56d23", "Above break"),
    ]

    for value, low, high, shift, color, label in slopes:
        axes[1].errorbar(
            data["breakpoint_m"] + shift,
            data[value],
            yerr=[data[value] - data[low], data[high] - data[value]],
            fmt="o",
            markersize=3,
            capsize=3,
            color=color,
            label=label,
        )

    axes[1].set_ylabel("Path-loss exponent")
    axes[1].legend(fontsize=7, frameon=False, loc="lower left")

    for ax in axes:
        ax.set_xlabel("Fixed breakpoint (m)")
        ax.set_xticks([5, 10, 15, 20])

    fig.savefig(output_dir / "dual_slope.png")
    plt.close(fig)


def main():
    output_dir.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.1,
            "savefig.dpi": 250,
        }
    )

    residual_plot()
    variogram_plot()
    qq_plot()
    dual_slope_plot()


if __name__ == "__main__":
    main()
