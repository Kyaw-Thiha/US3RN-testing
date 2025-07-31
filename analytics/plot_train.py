import pandas as pd
import plotly.express as px
from typing import Literal


KeyType = Literal["epoch", "batch"]

FILE_PATH = "logs/csv"

csv_file = "epoch_logs.csv"
plot_key: KeyType = "epoch"

# csv_file = "batch_logs.csv"
# plot_key: KeyType = "batch"


def plot_loss(df: pd.DataFrame, key: KeyType) -> None:
    """Plot standard loss curve with interactive hover."""
    fig = px.line(df, x=key, y="loss", title=f"Loss per {key.capitalize()}", markers=True)
    fig.update_layout(xaxis_title=key.capitalize(), yaxis_title="Loss")
    fig.show()


def plot_log_loss(df: pd.DataFrame, key: KeyType) -> None:
    """Plot log-scaled loss curve with hover info."""
    fig = px.line(df, x=key, y="loss", title=f"Loss per {key.capitalize()} (Log Scale)", markers=True, log_y=True)
    fig.update_layout(xaxis_title=key.capitalize(), yaxis_title="Loss (log scale)")
    fig.show()


def plot_loss_derivative(df: pd.DataFrame, key: KeyType) -> None:
    """Plot change in loss (ΔLoss) between steps."""
    df = df.copy()
    df["delta_loss"] = df["loss"].diff()
    fig = px.line(df.iloc[1:], x=key, y="delta_loss", title=f"Loss Delta Between {key.capitalize()}s", markers=True)
    fig.update_traces(line=dict(color="orange"))
    fig.update_layout(xaxis_title=key.capitalize(), yaxis_title="ΔLoss")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.show()


def plot_outliers(df: pd.DataFrame, key: KeyType) -> None:
    """Highlight points with unusually high loss values."""
    high_loss = df[df["loss"] > df["loss"].mean() + 2 * df["loss"].std()]
    print("Potential Outliers:\n", high_loss)

    fig = px.line(df, x=key, y="loss", title=f"{key.capitalize()} Loss with Outliers Highlighted", markers=True)
    fig.add_scatter(x=high_loss[key], y=high_loss["loss"], mode="markers", marker=dict(color="red", size=10), name="Outliers")
    fig.update_layout(xaxis_title=key.capitalize(), yaxis_title="Loss")
    fig.show()


if __name__ == "__main__":
    df = pd.read_csv(f"{FILE_PATH}/{csv_file}", parse_dates=["timestamp"])

    print("Data Summary:")
    print(df.describe())

    plot_loss(df, plot_key)
    plot_log_loss(df, plot_key)
    plot_loss_derivative(df, plot_key)
    plot_outliers(df, plot_key)
