import pandas as pd
import matplotlib.pyplot as plt

# 1. Read CSV (keep everything as strings for now)
df = pd.read_csv('metrics_log_multi_agent.csv', sep=';', dtype=str)

# 2. Parser function for Curiosity_Reward
def parse_cr(s):
    # if NaN or similar
    if pd.isna(s):
        return float('nan')
    # more than one dot → thousand separators
    if s.count('.') > 1:
        # remove all dots
        return float(s.replace('.', ''))
    # exactly one dot → decimal
    else:
        return float(s)

# 3. Convert columns
df['Curiosity_Reward'] = df['Curiosity_Reward'].apply(parse_cr)
df['Step'] = df['Step'].astype(int)

# 4. (Optional) focus on a certain range of steps to ignore outliers
# df = df[df['Step'] >= 100]
# df = df[df['Step'] <= 10000]

# 5. Downsampling for the plot (e.g., every 20th measurement)
df_ds = df.iloc[::20].copy()   # .copy() because we want to add a new column

# 6. Compute moving average: window size e.g. 50 points
#    You can adjust the window (e.g., 30, 100, etc.) depending on how much smoothing you want.
window_size = 50
df_ds['CR_smoothed'] = df_ds['Curiosity_Reward'].rolling(window=window_size, center=True).mean()

# 7. Plot: raw data + smoothed line
plt.figure(figsize=(10, 4))

# a) Raw data (downsampled)
plt.plot(
    df_ds['Step'],
    df_ds['Curiosity_Reward'],
    linewidth=0.8,
    alpha=0.5,
    color="blue",
    label='Raw Data (every 20th)'
)

# b) Smoothed curve
plt.plot(
    df_ds['Step'],
    df_ds['CR_smoothed'],
    color='red',
    linewidth=1.5,
    label=f'Moving Average (window={window_size})'
)

plt.xlabel('Step')
plt.ylabel('Curiosity Reward')
plt.title('Curiosity Reward Over Time')
plt.ylim(0.7, 1)
plt.legend()
plt.tight_layout()
plt.show()
