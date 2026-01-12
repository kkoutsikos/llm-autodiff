import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Data
try:
    df = pd.read_csv("training_results.csv")

    # 2. Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(df["Epoch"], df["Accuracy"], marker='o', label="Accuracy", color='green')
    plt.plot(df["Epoch"], df["Errors"] / 5, marker='x', linestyle='--', label="Error Rate", color='red')

    plt.title("Student Learning Curve (AdalFlow)")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    plt.show()

    print("\nLast Epoch Stats:")
    print(df.tail(1))

except FileNotFoundError:
    print("⚠️ No CSV found. Run the training script first!")