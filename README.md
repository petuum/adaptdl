# Artifact for Pollux OSDI 2021

This branch contains the artifact for the OSDI 2021 paper "Pollux: Co-adaptive
Cluster Scheduling for Goodput-Optimized Deep Learning", including:

- The main implementation of Pollux.
- Testbed experiment scripts (Sec 5.2).
- Cluster simulator (Sec 5.3).

## Pollux Implementation Files

Key files and modules:

- **adaptdl/adaptdl:** implementation of the PolluxAgent.
- **adaptdl/adaptdl/goodput.py:** Goodput function and throughput fitting.
- **adaptdl/adaptdl/torch/adascale.py:** AdaScale and square-root LR scaling.
- **adaptdl/adaptdl/torch/data.py:** Dynamic batch size and checkpoint-restart.
- **adaptdl/adaptdl/torch/parallel.py:** Throughput and statistical efficiency
  profiling during forward-backward passes.
- **sched/adaptdl_sched:** implementation of the PolluxSched.
- **sched/adaptdl_sched/policy/pollux.py:** PolluxSched cluster optimization.
- **simulator/** contains the implementation of the cluster simulator.

## Testbed Experiment Files

**benchmark/** contains the code we used to run our testbed experiments in
Section 5.2. The key files are:

- **benchmark/main.tf** Terraform file that contains the full configurations
  of the testbed cluster on AWS.
- **benchmark/models/** contains the implementations of each evaluated model
  described in Table 1.
- **benchmark/workloads/workload-6.csv** contains the manually-tuned job trace
  used to evaluate Pollux, Optimus+Oracle+TunedJobs, and Tiresias+TunedJobs.
  The GPUs and batch size configurations are ignored by the Pollux scheduler.
- **benchmark/workloads-realistic/workload-6.csv** contains the job trace used
  to evaluate Optimus+Oracle and Tiresias (Table 2).
- **benchmark/run_workload.py** submits jobs according to a workload trace.
- **benchmark/run_monitor.py** monitors and logs the cluster state during each
  cluster scheduling experiment.

## Cluster Simulator

`simulator/` contains Instructions for reproducing the experiments shown in
Section 5.3 and Section 5.3.2, and the analysis scripts to plot Figure 8.

Please see `simulator/README.md` for more details.
