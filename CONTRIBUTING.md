## Contributing to Adaptdl

**Thanks for taking the time to contribute!**

Refer to the following guidelines to contribute new functionality or bug fixes:

1. Use the following steps to develop your code:
    1. Clone the repository: `git clone https://github.com/petuum/adaptdl.git`
    2. Use `make deploy` with an active kubernetes cluster to deploy and test your changes
    3. If possible, run some example scripts from `tests/long-workload`. Ideally, have multiple Adaptdl jobs running simultaniously 
2. Use flake8 to lint the Python code: `flake8 adaptdl sched`.
3. Add pytest unit and/or integration tests for any new code you write. The testing files should exist in the same directory as the code files.
