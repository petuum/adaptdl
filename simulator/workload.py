import argparse
import numpy as np
import os
import pandas
import random

from datetime import datetime, timedelta

from applications import APPLICATIONS


def generate(num_jobs, start=0, duration=24, seed=0):
    trace_csv = os.path.join(os.path.dirname(__file__),
                             "traces", "philly.csv")
    trace = pandas.read_csv(trace_csv, parse_dates=["timestamp"])
    trace = trace[trace.duration >= 60]
    trace = trace[trace.gpu_time < 1000 * 3600]
    trace.timestamp -= timedelta(hours=start)
    trace = trace[trace.timestamp.dt.hour < duration]
    rng = random.Random(seed)
    rng2 = random.Random(seed + 3)
    rng3 = random.Random(seed + 4)
    sample = trace.sample(n=num_jobs, random_state=rng.randint(0, 1 << 32))
    records = []
    for row in sample.itertuples():
        hour = row.timestamp.hour
        minute = row.timestamp.minute
        second = row.timestamp.second
        rec = {"time": hour * 3600 + minute * 60 + second}
        num_gpus = row.num_gpus
        if row.gpu_time < 1 * 3600:
            rec["application"] = rng.choice(["cifar10", "ncf"])
        elif row.gpu_time < 10 * 3600:
            rec["application"] = "deepspeech2"
        elif row.gpu_time < 100 * 3600:
            rec["application"] = "yolov3"
            subset = trace[trace.duration <= 24 * 3600]
            subset = subset[subset.gpu_time >= 10 * 3600]
            subset = subset[subset.gpu_time < 100 * 3600]
            num_gpus = rng3.choice(subset.num_gpus.to_list())
        else:
            rec["application"] = "imagenet"
            subset = trace[trace.duration <= 24 * 3600]
            subset = subset[subset.gpu_time >= 100 * 3600]
            num_gpus = rng3.choice(subset.num_gpus.to_list())
        rec["num_replicas"], rec["batch_size"], _ = rng.choice(
            APPLICATIONS[rec["application"]].get_configurations(0.5, 0.8))
        if rec["application"] == "deepspeech2" and rng2.randint(0, 1):
            # Change half of the deepspeech2 jobs to bert jobs. Use a different
            # random number generator to avoid affecting the rest of the jobs.
            rec["application"] = "bert"
            rec["num_replicas"], rec["batch_size"], _ = rng2.choice(
                APPLICATIONS[rec["application"]].get_configurations(0.5, 0.8))
        #rec["num_replicas"] = num_gpus
        #rec["batch_size"] = APPLICATIONS[rec["application"]].init_batch_size * num_gpus
        records.append(rec)
    records.sort(key=lambda v: v["time"])
    for idx, rec in enumerate(records):
        rec["name"] = "{}-{}".format(rec["application"], idx)
    return pandas.DataFrame(records, columns=("name", "time", "application",
                                              "num_replicas", "batch_size"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", type=int, default=2,
                        help="starting hour")
    parser.add_argument("-d", "--duration", type=int, default=8,
                        help="total number of workload hours")
    parser.add_argument("-n", "--num-jobs", type=int, default=160,
                        help="total number of jobs")
    parser.add_argument("-o", "--output", type=str,
                        help="path to output the workload")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed")
    args = parser.parse_args()
    workload = generate(args.num_jobs, start=args.start,
                        duration=args.duration, seed=args.seed)
    csv = workload.set_index("name").to_csv(args.output)
    if csv:
        print(csv)
    print(workload.groupby(["application", "num_replicas", "batch_size"])
          .size().reset_index(name="count"))
