# Input: cleaned runs in df, including date and rejected (0 or 1).
# Output: a plot of the monthly rejection percentage; the dashed line marks the 2024 test set.
runs_over_time = df.copy()
runs_over_time["month"] = pd.to_datetime(runs_over_time["date"]).dt.to_period("M").dt.to_timestamp()
monthly = runs_over_time.groupby("month")["rejected"].agg(rejection_rate="mean", runs="size")

ax = (monthly["rejection_rate"] * 100).plot(figsize=(11, 4), marker="o", markersize=3)
ax.axvline(pd.Timestamp("2024-01-01"), color="gray", linestyle="--", label="2024 test starts")
ax.set(title="Monthly rejection rate", xlabel="Submission month", ylabel="Rejected runs (%)")
ax.legend()
plt.tight_layout()
plt.show()


# Input: numeric columns from the training data.
# Output: covariance matrix heatmap showing how numeric variables change together.
covariance = train[NUMERIC].cov()

plt.figure(figsize=(8, 6))
plt.imshow(covariance, cmap="coolwarm")
plt.colorbar(label="Covariance")
plt.xticks(range(len(NUMERIC)), NUMERIC, rotation=45, ha="right")
plt.yticks(range(len(NUMERIC)), NUMERIC)
plt.title("Covariance matrix: training data")
plt.tight_layout()
plt.show()