import pandas as pd
import matplotlib.pyplot as plt


QIA_metrics = pd.read_csv("QIA_metrics.csv", index_col=0)
#print(QIA_metrics.head())

QIA_metrics["folder"] = pd.to_datetime(QIA_metrics["folder"], format="%Y-%m-%d")
numeric_columns = QIA_metrics.select_dtypes(include=['float64', 'int64']).columns
grouped = QIA_metrics.groupby("folder")[numeric_columns].agg(["mean", "std"])

for col in numeric_columns:
    means=grouped[(col, "mean")]
    stds = grouped[(col, "std")]

    plt.figure(figsize=(10, 6))
    plt.errorbar(means.index, means, yerr=stds, fmt='o', capsize=5)
    plt.title(f"{col}")
    plt.xlabel("Date")
    plt.ylabel(col)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


means = grouped.xs("mean", axis=1, level=1)
corr = means.corr(method="pearson")  

print(corr)
plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.tight_layout()
plt.show()