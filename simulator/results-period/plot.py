import json
import matplotlib.pyplot
import pandas
import seaborn


period_list = ["30s", "1m", "2m", "4m", "8m"]
period_seconds = [30, 60, 120, 240, 480]

records = []
for period, seconds in zip(period_list, period_seconds):
  with open("pollux-{}/summary.json".format(period)) as f:
    summary = json.load(f)
  for workload, jcts in summary["jcts"].items():
    records.append({
      "workload": workload,
      "hue": "hue",
      "period": period,
      "seconds": seconds,
      "jct": sum(jcts.values()) / len(jcts),
    })

df = pandas.DataFrame.from_records(records)

df = df.groupby(["period", "hue", "workload"]).mean().reset_index()
df = df.sort_values("seconds")

df.jct /= 3600

seaborn.set_style("whitegrid")
seaborn.set_style("whitegrid", {"font.family": "serif"})
matplotlib.rcParams['legend.labelspacing'] = 0.4
matplotlib.rcParams['legend.columnspacing'] = -5
fig, ax1 = matplotlib.pyplot.subplots(figsize=(2.5, 2))
#seaborn.lineplot(x=df.seconds, y=df.jct, ci=None, ax=ax1, legend=False)
pl = seaborn.barplot(x=df.period, y=df.jct, hue=df.hue, ci=95, errwidth=1.5, capsize=0.25, ax=ax1)
pl.legend_.remove()
for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x() + p.get_width() / 2, height + 0.15, '{:1.2f}'.format(height), ha="center", fontsize=9)
#ax1.set_xlim(0, None)
ax1.set_ylim(0, 1.2)
#ax1.lines[0].set_marker("o")
#ax1.lines[0].set_markersize(5)
#ax1.lines[0].set_linestyle("-")
#ax1.set_xticks([0, 100, 200])
ax1.set_yticks([0.0, 0.5, 1.0])
#ax1.set_xticklabels(["0.5x", "1.0x", "1.5x", "2.0x"])
#ax1.set_yticks([0, 1, 2, 3])
ax1.set_xlabel("Scheduling Interval")
ax1.set_ylabel("Avg JCT (hours)")
#ax1.legend(handles=ax1.lines, labels=["Pollux (p = -1)", "Optimus+Oracle+TunedJobs", "Tiresias+TunedJobs"],
#           fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.0),  ncol=2)

fig.tight_layout()
fig.savefig("simulator-period.pdf")
matplotlib.pyplot.show()
