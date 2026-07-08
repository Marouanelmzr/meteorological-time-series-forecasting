import matplotlib.pyplot as plt

def plot_target_distribution(df):
    counts = df["has_gust"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(["No Gust", "Gust"], counts.values )
    ax.set_title("Target Distribution")
    ax.set_ylabel("Number of samples")

    plt.tight_layout()

    return fig