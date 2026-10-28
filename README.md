# LBM

A D2Q9 lattice Boltzmann solver for flow past a cylinder, in C++ with
OpenMP. It was a pair project at UPC, written up in Towards Data Science as
[The Fluid Simulator That Doesn't Solve the Fluid Equations](https://towardsdatascience.com/the-fluid-simulator-that-doesnt-solve-the-fluid-equations/).
The benchmarks on MareNostrum 5 are on the
[project page](https://feal-ca.github.io/work/lattice-boltzmann/).

## Files

- `lbm.cpp` is the serial solver, with τ fixed at 0.6.
- `lbm_parallel.cpp` is the same solver with OpenMP. τ is its first
  argument, 0.6 if you leave it out.
- `lbm_numpy.py` is a NumPy version that saves velocity magnitude and
  vorticity plots as PNGs.
- `obstacle.py` is a small Tkinter tool for drawing an obstacle on the
  400×400 grid and exporting it to `obstacle_map.txt`. None of the solvers
  read that file yet; they all use the cylinder.

## Running it

```sh
g++ -O3 -fopenmp lbm_parallel.cpp -o lbm_parallel
OMP_NUM_THREADS=8 ./lbm_parallel 0.55
```

It runs 10,000 steps and writes an ASCII VTK file every 50 (`lbm_0.vtk`,
`lbm_50.vtk` and so on) with the velocity, the density and the solid mask.
ParaView opens them as a time series. Each one is about 5.5 MB, so a full
run takes about 1.1 GB of disk.

The serial version builds without OpenMP: `g++ -O3 lbm.cpp -o lbm`.

The NumPy version needs NumPy and Matplotlib:

```sh
python lbm_numpy.py --tau 0.55 --steps 5000 --out_every 500 --save_dir lbm_output
```

Those are also the defaults. With `--save_dir ""` it shows the plots
instead of saving them.

## The setup

The grid is 400×400 cells. The cylinder has a radius of 40 cells and sits a
quarter of the way along, 5 cells above the centerline. The inlet on the left is a Zou-He velocity boundary at
u = 0.1 and the outlet copies the last column. The top and bottom walls and
the cylinder use bounce-back, and collision is BGK.

The relaxation time sets the viscosity, ν = (τ − 0.5)/3, and with it the
Reynolds number on the cylinder's diameter. At τ = 0.75 the wake is steady
and Re ≈ 96. At τ = 0.55, Re ≈ 480 and a vortex street forms. τ has to stay
above 0.5.
