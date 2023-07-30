import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from tqdm import tqdm

save_fig_root = "/home/giang/aws_hackathon/assets/figures"
num_data_points = 1000


def parse_data(json_path):
    global num_data_points, save_fig_root

    with open(json_path, 'r') as f:
        data = json.load(f)

    save_csv = []
    for d in tqdm(data, total=len(data)):
        if len(d) == 0:
            continue
        try:
            record = d[0]
            time_stamp = record[0]
            time_stamp = pd.to_datetime(time_stamp, unit='ms')
            save_csv.append({
                "Timestamp": time_stamp,
                "Open": float(record[1]),
                "High": float(record[2]),
                "Low": float(record[3]),
                "Close": float(record[4]),
                "Volume": float(record[5]),
            })
        except Exception:
            print(d)
            exit()
    df = pd.DataFrame(save_csv[-num_data_points:])

    for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
        plt.figure(figsize=(12, 6))
        ax = sns.lineplot(data=df, x='Timestamp', y=column)
        plt.xlabel('Time')
        plt.title(f'{column} over Time')

        # Customize the y-axis ticks and labels
        yticks = ax.get_yticks()
        ax.set_yticks(yticks[::3])
        ax.set_yticklabels([f'{y:,.0f}' for y in yticks[::3]])

        plt.savefig(os.path.join(save_fig_root, f'{column.lower()}_plot.png'))
        plt.close()


if __name__ == "__main__":
    parse_data("/home/giang/aws_hackathon/data/sol_from_begin.json")
