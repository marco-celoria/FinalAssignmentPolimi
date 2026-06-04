In the 07-Cooling folder there is the sequential source code 'Cooling2025.c' and scripts for compiling 
and running on Leonardo (with few changes on any other Linux platform).
Tests have been carried out with GNU compiler, but other C compilers should do as well.

The parallelisation benchmarks should be run with the original 'Cooling.inp' file, but during 
the optimization process you could use 'CoolingLight.inp' in which the parameter values are reduced to
Xdots=Ydots=1000, MaxIters=4000, Steps=24. 
You can run the small case by launching ./Cooling2025.exe CoolingLight.inp

In order to check parallelization correctness the numerical values 'min, mean, max, std.dev.'
written by the original serial code in 'Statistics.csv' should match those written by the optimized program.

The original program may produce a lot of .ppm image files also, with which a movie can be visualized using
the utility program Movie.py.
You may choose to avoid generating these files by changing to 0 the last value in 'Cooling.inp'. 
If so please report your decision when you show benchmark timings.

Would you please read the comments in the source code for further instructions.

Would you please let me know about code issues by writing to m.cremonesi@cineca.it 
