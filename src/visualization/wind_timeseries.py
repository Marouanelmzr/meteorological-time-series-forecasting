import matplotlib.pyplot as plt
import pandas as pd

def plot_wind_timeseries(df):

    temp = df.copy()
    temp["time"] = pd.to_datetime(temp["time"])
    temp = temp.sort_values("time")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(temp["time"], temp["wind_final"])
    ax.set_xlabel("Time")
    ax.set_ylabel("Wind Speed (m/s)")
    ax.set_title("Wind Speed Timeseries")
    
    plt.tight_layout()
    
    return fig