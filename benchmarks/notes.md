# Note about benchmarks.

Every folder includes a system with a prmtop file (`complex.prmtop`), a `complex.cpin` file, and a minimized structure (`min.pdb`) which are used to construct the system.

To run the benchmark, invoke the script [`benchmarker.py`](benchmarker.py) in the subdirectory of the system. See also [`run_benchmark.sh`](run_benchmark.sh).

Settings that are important to note:
* 5000  dynamics/titration cycles to run
* 500 time steps of dynamics per iteration
    * 1.0 femtoseconds time step
* The number of titration trials was determined by the total number of titratable groups in the protein.
  * DHFR: 30
  * Abl: 39
  * LDHA: 172
* I used `openmm-dev                7.0.0.dev0               py35_0` on linux 64
* The data was collected on our local `src` dev box, using the CUDA platform on our GeForce GTX TITAN.

For further details, see [`benchmarker.py`](benchmarker.py).

The file `benchmark.txt` includes the average cost per time step or average cost per titration trial for each iteration (500 time steps + `N` titration trials).
The corresponding `summmary.txt` file has a summary of the entire benchmark.

Also included are cProfile files (`benchmark.prof`) that can be viewed using `pstats` :

```
python -m pstats bench.prof
```

Or a visualization tool of your choosing. I used [`pyprof2calltree`](https://pypi.python.org/pypi/pyprof2calltree), with this command:

```
pyprof2calltree -i benchmark.prof -k
```

Every directory has a `benchmark.png` file in it, with a call graph of the data in `benchmarks.prof`, and a callgraph.dot(.ps) file with the callgraph.

## Results

dhfr-implicit : 19.25 ns per day

dhfr-explicit : 11.48 ns per day

abl-implicit: 11.45 ns per day

abl-explicit : 8.66 ns per day

ldha-implicit: roughly 0.625 ns per day

ldha-explciit: roughly 1.3 ns per day
