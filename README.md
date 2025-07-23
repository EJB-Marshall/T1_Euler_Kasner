# T1_Euler_Kasner
Code to evolve the T1 symmetric Euler equations on Minkowski and fixed Kasner spacetimes.

Mathematica files contain derivations for source terms and characteristic speeds.

The Euler_Kasner.py file takes the following command line arguments:

Nx - Number of cells in the x direction<br/>
Ny - Number of cells in the y direction<br/>
K - The sound speed parameter<br/>
d - (optional) Plots the solution using Matplotlib<br/>

For example, a simulation with 100 cells along each axis and K = 0.5<br/>
could be run by typing:<br/>

$ python3 -O Euler_Kasner.py -Nx 100 -Ny 100 -K 0.5

