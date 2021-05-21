# Pollux Cluster Simulator

This directory contains the code used for the simulator-based experiments in
Section 5.3 of the paper. In particular, you should be able to use this code
to run the simulator on the same 8 worklaods we used for the simulator study,
check the fidelity of the simulator as describe in Section 5.3 of the paper,
and to reproduce the experiments described by Figure 8 in the paper.

The contents are summarized as follows:

- **workloads/** contains the 8 different workloads used for evaluation.
- **workloads-0.5/**, **workloads-1.5/**, and **workloads-2.0/** contain the
  0.5x, 1.5x, and 2.0x "relative job load" workloads used for Figure 8b.
- **traces/** contains the system throughput and statistical efficiency
  measurements collected according to the "Simulator construction" paragraph.
- **applications.py** contains the code that parses the collected traces of
  each training job type, as well as helpers to interpolate to GPU placement
  and batch size configurations which were not directly measured.
- **optimus.py** and **tiresias.py** contain implementations of
  the Optimus+Oracle and Tiresias scheduling policies, respectively.
- **pollux.py**, **speedup.py**, **goodput.py**, and **utils.py** contain the
  implementation of the Pollux scheduling policy and Pollux agent.
- **workload.py** is the script that was used to generate the evaluation
  workloads.
- **results-jobload/** contains the raw results for Figure 8a.
- **results-period/** contains the raw results for Figure 8b.
- **results-inter/** contains the raw results for Figure 8c.

## Getting Started

We suggest using a fresh virtualenv or conda environment to install the
dependencies:

```
$ conda create -n pollux python=3.8
$ conda activate pollux
$ python3 -m pip install -r requirements.txt
```

## Checking Simulator Fidelity

Use the following command to run the Pollux policy on Workload #6, which is the
one used for the Pollux, Optimus+TunedJobs, and Tiresias+TunedJobs testbed
experiments in Section 5.2. The simulator should print out logs for every 60s
of simulated time, along with some information on the active and running jobs.
At the end, it will print out the average completion time of the jobs in the
workload. 

```
$ python3 simulator.py --policy pollux workloads/workload-6.csv
...
...
...
Average JCT: 2446.13125
```

Running the Pollux policy may take up to 30mins depending on the speed of the
CPU used to run the simulator. Your resulting average JCT may also be slightly
different than the above due to randomness in the Pollux scheduling policy.

Running the Optimus+Oracle+TunedJobs policy:

```
$ python3 simulator.py --policy optimus workloads/workload-6.csv
...
...
...
Average JCT: 5301.875
```

Running the Tiresias+TunedJobs policy:

```
$ python3 simulator.py --policy tiresias workloads/workload-6.csv
...
...
...
Average JCT: 3469.69375
```

Pollux should have a roughly 48% lower average JCT than Optimus, and roughly
32% lower JCT than Tiresias, as reported in Section 5.3, within a reasonable
(plus or minus 5%) margin of error.

## Sensitivity to Job Load

Our raw results for Figure 8a are provided in `results-jobload`. To reproduce
this experiment, create a new directory and copy over the `plot.py` script. You
will need to run each experiment to re-create the directory structure.

```
$ mkdir reproduce-jobload
$ cp results-jobload/plot.py reproduce-jobload
```

To run the Pollux policy on all 8 workloads in `workloads`:

```
$ mkdir reproduce-jobload/pollux-1.0
$ python3 simulator.py --policy pollux --output reproduce-jobload/pollux-1.0 workloads/
```

The above command will run all 8 worklaods under `workloads` in parallel and
write a summary of the results to `reproduce-jobload/pollux-1.0`. It's
recommended to run it on a machine that has at least 8 CPUs.

Now, run similar experiments for the 0.5x, 1.5x, and 2.0x workloads, e.g.

```
$ mkdir reproduce-jobload/pollux-0.5
$ python3 simulator.py --policy pollux --output reproduce-jobload/pollux-0.5 workloads-0.5/
...
$ mkdir reproduce-jobload/pollux-1.5
$ python3 simulator.py --policy pollux --output reproduce-jobload/pollux-1.5 workloads-1.5/
...
$ mkdir reproduce-jobload/pollux-2.0
$ python3 simulator.py --policy pollux --output reproduce-jobload/pollux-2.0 workloads-2.0/
...
```

Similarly, repeat the above experiments for the `optimus` and `tiresias` policies.

Lastly, plot the results:

```
$ cd reproduce-jobload
$ python3 plot.py
```

## Impact of Scheduling Interval

Our raw results for Figure 8b are provided in `results-period`. To reproduce
this experiment, create a new directory and copy over the `plot.py` script. You
will need to run each experiment to re-create the directory structure.

```
$ mkdir reproduce-period
$ cp results-period/plot.py reproduce-period
```

The scheduling interval can be set using the `--interval` flag (in seconds):

```
$ mkdir reproduce-period/pollux-1m
$ python3 simulator.py --policy pollux --interval 60 --output reproduce-period/pollux-1m workloads/
```

Next, repeat the above experiment for the 30s, 2m, 4m, and 8m intervals, and
plot the results:

```
$ cd reproduce-period
$ python3 plot.py
```

## Impact of Interference Avoidance

Our raw results for Figure 8c are provided in `results-inter`. To reproduce
this experiment, create a new directory and copy over the `plot.py` script. You
will need to run each experiment to re-create the directory structure.

```
$ mkdir reproduce-inter
$ cp results-inter/plot.py reproduce-inter
```

The slowdown due to interference can be set using the `--interference` flag:

```
$ mkdir reproduce-inter/avoid
$ python3 simulator.py --policy pollux --interference 0.0 --output reproduce-inter/avoid workloads/
...
$ mkdir reproduce-inter/avoid25
$ python3 simulator.py --policy pollux --interference 0.25 --output reproduce-inter/avoid25 workloads/
...
$ mkdir reproduce-inter/avoid50
$ python3 simulator.py --policy pollux --interference 0.5 --output reproduce-inter/avoid50 workloads/
...
```

The three experiments above run Pollux with interference avoidance enabled with
different levels of artificial network interference injected (0% slowdown, 25%
slowdown, and 50% slowdown for each job that interferes with another job).

Since artifical interference is only injected for distributed jobs which share
a same node with another distributed job (a situation which is should not occur
using Pollux with interference avoidance enabled), our analysis/plotting script
assumes all three experiments would be the same and only uses the results in
the `reproduce-inter/avoid` directory. You should check this assumption that
all three experiments above produce similar results.

Next, to disable interference avoidance, you will need to modify `pollux.py` by
commenting out lines 387 to 390. Then, repeat the three runs above:

```
$ mkdir reproduce-inter/noavoid
$ python3 simulator.py --policy pollux --interference 0.0 --output reproduce-inter/noavoid workloads/
...
$ mkdir reproduce-inter/inter25
$ python3 simulator.py --policy pollux --interference 0.25 --output reproduce-inter/inter25 workloads/
...
$ mkdir reproduce-inter/inter50
$ python3 simulator.py --policy pollux --interference 0.5 --output reproduce-inter/inter50 workloads/
...
```

Lastly, plot the results:

```
$ cd reproduce-inter
$ python3 plot.py
```
