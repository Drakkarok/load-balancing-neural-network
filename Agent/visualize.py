
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_metrics(csv_path="Models/metrics/training_metrics.csv"):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Create Plots Directory
    output_dir = os.path.join(os.path.dirname(csv_path), "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Total Reward over Episodes
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['total_reward'], label='Total Reward', alpha=0.6)
    # Add rolling average
    plt.plot(df['episode'], df['total_reward'].rolling(window=20).mean(), label='Rolling Avg (20)', color='red')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "reward_plot.png"))
    print(f"Saved {output_dir}/reward_plot.png")
    
    # 2. Gini Coefficient (Fairness)
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['gini_coefficient'], label='Gini Coefficient', color='orange', alpha=0.6)
    plt.plot(df['episode'], df['gini_coefficient'].rolling(window=20).mean(), label='Rolling Avg (20)', color='darkorange')
    plt.xlabel("Episode")
    plt.ylabel("Gini Coefficient (Lower is Better)")
    plt.title("Load Fairness (Gini) over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "gini_plot.png"))
    print(f"Saved {output_dir}/gini_plot.png")

    # 3. Optimal Decision Rate
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['optimal_decision_rate'], label='Optimal Decisions %', color='green', alpha=0.6)
    plt.plot(df['episode'], df['optimal_decision_rate'].rolling(window=20).mean(), label='Rolling Avg (20)', color='darkgreen')
    plt.xlabel("Episode")
    plt.ylabel("Optimal Decisions (%)")
    plt.title("Agent Accuracy over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
    print(f"Saved {output_dir}/accuracy_plot.png")

    print("\nVisualization Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="Models/metrics/training_metrics.csv", help="Path to metrics CSV")
    args = parser.parse_args()
    
    plot_metrics(args.csv)
