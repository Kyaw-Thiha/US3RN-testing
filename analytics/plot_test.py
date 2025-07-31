import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Literal


KeyType = Literal["epoch", "batch"]

FILE_PATH = "logs/csv"

csv_file = "test.csv"
plot_key: KeyType = "epoch"

# csv_file = "batch_logs.csv"
# plot_key: KeyType = "batch"


def plot_test(df: pd.DataFrame, key: KeyType) -> None:
    """
    Plot normalized PSNR and raw SSIM curves with lowest point highlighted

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing at least the columns: `key`, `psnr`, `ssim`.
        key : Literal["epoch", "batch"]
            Column to use as x-axis.
    """
    # Normalize PSNR between 0 and 1
    psnr_min, psnr_max = df["psnr"].min(), df["psnr"].max()
    df["psnr_norm"] = (df["psnr"] - psnr_min) / (psnr_max - psnr_min)

    # Identify lowest points
    psnr_max_idx = df["psnr"].idxmax()
    ssim_max_idx = df["ssim"].idxmax()

    psnr_color = "#FFB347"
    ssim_color = "#70D6FF"

    fig = go.Figure()

    # PSNR line
    fig.add_trace(
        go.Scatter(
            x=df[key],
            y=df["psnr_norm"],
            mode="lines+markers",
            name="PSNR",
            line=dict(color=psnr_color),
        )
    )

    # Highlight PSNR max
    fig.add_trace(
        go.Scatter(
            x=[df[key][psnr_max_idx]],
            y=[df["psnr_norm"][psnr_max_idx]],
            mode="markers+text",
            name="Max PSNR",
            marker=dict(color=psnr_color, size=10, symbol="circle-open"),
            text=[f"Max: {df['psnr'][psnr_max_idx]:.2f}"],
            textposition="bottom center",
            showlegend=False,
        )
    )

    # SSIM line
    fig.add_trace(
        go.Scatter(
            x=df[key],
            y=df["ssim"],
            mode="lines+markers",
            name="SSIM",
            line=dict(color=ssim_color),
        )
    )

    # Highlight SSIM max
    fig.add_trace(
        go.Scatter(
            x=[df[key][ssim_max_idx]],
            y=[df["ssim"][ssim_max_idx]],
            mode="markers+text",
            name="Max SSIM",
            marker=dict(color=ssim_color, size=10, symbol="circle-open"),
            text=[f"Max: {df['ssim'][ssim_max_idx]:.4f}"],
            textposition="bottom center",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"PSNR & SSIM per {key.capitalize()}",
        xaxis_title=key.capitalize(),
        yaxis_title="Metric Value",
        legend=dict(x=0.01, y=0.99),
    )

    fig.show()


def plot_psnr(df: pd.DataFrame, key: KeyType) -> None:
    """Plot PSNR curve with interactive hover."""
    fig = px.line(df, x=key, y="psnr", title=f"PSNR per {key.capitalize()}", markers=True)
    fig.update_layout(xaxis_title=key.capitalize(), yaxis_title="PSNR (dB)")
    fig.show()


def plot_ssim(df: pd.DataFrame, key: KeyType) -> None:
    """Plot SSIM curve with interactive hover."""
    fig = px.line(df, x=key, y="ssim", title=f"SSIM per {key.capitalize()}", markers=True)
    fig.update_layout(xaxis_title=key.capitalize(), yaxis_title="SSIM")
    fig.show()


if __name__ == "__main__":
    df = pd.read_csv(f"{FILE_PATH}/{csv_file}")

    print("Data Summary:")
    print(df.describe())

    plot_test(df, plot_key)
    plot_psnr(df, plot_key)
    plot_ssim(df, plot_key)
