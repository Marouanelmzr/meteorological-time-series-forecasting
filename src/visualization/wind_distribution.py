import matplotlib.pyplot as plt

def plot_wind_distribution(df):
    
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(df["wind_final"], bins=30)
    ax.set_title("Wind Speed Distribution")
    ax.set_xlabel("Wind Speed")
    ax.set_ylabel("Frequency")

    plt.tight_layout()
    
    return fig