import json
import matplotlib.pyplot
import pandas
import seaborn


policy_list = ["pollux", "optimus", "tiresias"]
jobload_list = [0.5, 1.0, 1.5, 2.0]

records = []
for policy in policy_list:
  for jobload in jobload_list:
    with open("{}-{}/summary.json".format(policy, jobload)) as f:
      summary = json.load(f)
    for workload, jcts in summary["jcts"].items():
      path = f"../workloads/{workload}.csv" if jobload == 1.0 else f"../workloads-{jobload}/{workload}.csv"
      df = pandas.read_csv(path)
      records.append({
        "workload": workload,
        "policy": policy,
        "jobload": jobload,
        "jct": sum(jcts.values()) / len(jcts),
        "makespan": max(row.time + jcts[row.name] for row in df.itertuples()),
      })

df = pandas.DataFrame.from_records(records)

df = df.groupby(["policy", "jobload", "workload"]).mean().reset_index()

df.jct /= 3600
df.makespan /= 3600

print(df)

seaborn.set_style("whitegrid")
seaborn.set_style("whitegrid", {"font.family": "serif"})
matplotlib.rcParams['legend.labelspacing'] = 0.4
matplotlib.rcParams['legend.columnspacing'] = -5
fig, (ax1, ax2) = matplotlib.pyplot.subplots(1, 2, figsize=(6, 2))
seaborn.lineplot(x=df.jobload, y=df.jct, hue=df.policy, hue_order=policy_list, ci=95, ax=ax1, legend=False)
ax1.set_ylim(0, 3)
ax1.lines[0].set_marker("o")
ax1.lines[1].set_marker("v")
ax1.lines[2].set_marker("X")
ax1.lines[0].set_markersize(5)
ax1.lines[1].set_markersize(7)
ax1.lines[2].set_markersize(7)
ax1.lines[0].set_linestyle("-")
ax1.lines[1].set_linestyle("--")
ax1.lines[2].set_linestyle(":")
ax1.set_xticks([0.5, 1.0, 1.5, 2.0])
ax1.set_xticklabels(["0.5x", "1.0x", "1.5x", "2.0x"])
ax1.set_yticks([0, 1, 2, 3])
ax1.set_xlabel("Relative Job Load")
ax1.set_ylabel("Avg JCT (hours)")
seaborn.lineplot(x=df.jobload, y=df.makespan, hue=df.policy, hue_order=policy_list, ci=95, ax=ax2, legend=False)
ax2.set_ylim(0, None)
ax2.lines[0].set_marker("o")
ax2.lines[1].set_marker("v")
ax2.lines[2].set_marker("X")
ax2.lines[0].set_markersize(5)
ax2.lines[1].set_markersize(7)
ax2.lines[2].set_markersize(7)
ax2.lines[0].set_linestyle("-")
ax2.lines[1].set_linestyle("--")
ax2.lines[2].set_linestyle(":")
ax2.set_xticks([0.5, 1.0, 1.5, 2.0])
ax2.set_xticklabels(["0.5x", "1.0x", "1.5x", "2.0x"])
#ax2.set_yticks([0, 1, 2, 3])
ax2.set_xlabel("Relative Job Load")
ax2.set_ylabel("Makespan (hours)")
fig.legend(handles=ax1.lines, labels=["Pollux (p = -1)", "Optimus+Oracle\n+TunedJobs", "Tiresias\n+TunedJobs"],
           fontsize=9, loc=5, bbox_to_anchor=(0.9, 0.6),  ncol=1)

fig.tight_layout(rect=[0,0,0.65,1])
fig.savefig("simulator-jobload.pdf")
matplotlib.pyplot.show()
