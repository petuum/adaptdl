import json
import matplotlib.pyplot
import pandas
import seaborn


records = []

with open("avoid/summary.json") as f:
  summary_avoid = json.load(f)
  for workload, jcts in summary_avoid["jcts"].items():
    for jct in jcts.values():
      records.append({
        "workload": workload,
        "avoidance": "on",
        "interference": "0%",
        "jct": jct,
      })
      records.append({
        "workload": workload,
        "avoidance": "on",
        "interference": "25%",
        "jct": jct,
      })
      records.append({
        "workload": workload,
        "avoidance": "on",
        "interference": "50%",
        "jct": jct,
      })

with open("noavoid/summary.json") as f:
  summary_noavoid_00 = json.load(f)
  for workload, jcts in summary_noavoid_00["jcts"].items():
    for jct in jcts.values():
      records.append({
        "workload": workload,
        "avoidance": "zoff",
        "interference": "0%",
        "jct": jct,
      })

with open("inter25/summary.json") as f:
  summary_noavoid_25 = json.load(f)
  for workload, jcts in summary_noavoid_25["jcts"].items():
    for jct in jcts.values():
      records.append({
        "workload": workload,
        "avoidance": "zoff",
        "interference": "25%",
        "jct": jct,
      })

with open("inter50/summary.json") as f:
  summary_noavoid_50 = json.load(f)
  for workload, jcts in summary_noavoid_50["jcts"].items():
    for jct in jcts.values():
      records.append({
        "workload": workload,
        "avoidance": "zoff",
        "interference": "50%",
        "jct": jct,
      })

df = pandas.DataFrame.from_records(records)

df = df.groupby(["interference", "avoidance", "workload"]).mean().reset_index()

print(df)

#df.jct /= df[df.avoidance == "on"].jct.mean()
df.jct /= 3600

seaborn.set_style("whitegrid")
seaborn.set_style("whitegrid", {"font.family": "serif"})
fig, ax = matplotlib.pyplot.subplots(figsize=(2.5, 2.1))
seaborn.barplot(x="interference", y="jct", hue="avoidance", ci=95, errwidth=1.2, capsize=0.12, ax=ax, data=df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, height + 0.3, '{:1.2f}'.format(height), ha="center", fontsize=7.5) 
hatches = ['///', '\\\\\\']
for i, bar in enumerate(ax.patches):
    bar.set_hatch(hatches[i // 3])
ax.set_ylim(0, 1.6)
ax.set_xlabel("Interference slowdown")
ax.set_ylabel("Avg JCT (hours)")
print(ax.patches)
ax.legend(handles=[ax.patches[0], ax.patches[3]], labels=["Avoidance enabled", "Avoidance disabled"], bbox_to_anchor=(0.5, 1.05), loc="lower center", ncol=1, fontsize=9)

fig.tight_layout()
fig.savefig("interference.pdf")
matplotlib.pyplot.show()
