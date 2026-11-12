import matplotlib.pyplot as plt
import pandas as pd

from candidates import CANDIDATES
from paths import CLEAN_RUNS, CANDIDATE_RUNS_PLOT


runs = pd.read_csv(CLEAN_RUNS, usecols=["gameName", "status"])
games = list(dict.fromkeys(CANDIDATES))

counts = runs[runs.gameName.isin(games)].groupby(["gameName", "status"]).size()
counts = counts.unstack(fill_value=0).reindex(games, fill_value=0)
counts = counts.reindex(columns=["verified", "rejected", "new"], fill_value=0)

print("Candidate games missing from the runs data:")
print(counts.index[counts.sum(axis=1) == 0].tolist())
print(counts)

counts.plot.barh(stacked=True, figsize=(12, 10))
plt.xlabel("Runs submitted")
plt.ylabel("Game")
plt.title("Runs submitted by candidate game and outcome")
plt.legend(title="Outcome")
plt.tight_layout()
plt.savefig(CANDIDATE_RUNS_PLOT, dpi=150)
plt.show()
