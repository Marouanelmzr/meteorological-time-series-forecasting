import matplotlib.pyplot as plt

def plot_wind_by_target(df):

    fig, ax = plt.subplots(figsize=(8, 6))

    data = [
        df[df["has_gust"] == 0]["wind_final"],
        df[df["has_gust"] == 1]["wind_final"]
    ]

    ax.boxplot(data, tick_labels=["No Gust", "Gust"])

    ax.set_title("Wind Speed by Target")
    ax.set_ylabel("Wind Speed")

    plt.tight_layout()

    return fig