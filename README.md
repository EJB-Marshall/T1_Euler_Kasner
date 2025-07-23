# T1_Euler_Kasner
Code to evolve the T1 symmetric Euler equations on Minkowski and fixed Kasner spacetimes.

Mathematica files contain derivations for source terms and characteristic speeds.

The Euler_Kasner.py file takes the following command line arguments:

Nx - Number of cells in the x direction
Ny - Number of cells in the y direction
K - The sound speed parameter
d - (optional) Plots the solution using Matplotlib

For example, a simulation with 100 cells along each axis and K = 0.5
could be run by typing:

$ python3 -O Euler_Kasner.py -Nx 100 -Ny 100 -K 0.5

