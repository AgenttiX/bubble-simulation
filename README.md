# bubble-simulation

A live, interactive GPU lattice simulation of **bubble nucleation and growth in
a first-order cosmological phase transition**, built for teaching.

Bubbles of a new phase are seeded in a hot relativistic plasma, their walls
accelerate outwards under the released vacuum energy, they stir and shock the
fluid, and they collide — leaving the box ringing with the sound waves that are
the dominant source of gravitational waves from such a transition. You can fly
through all of it while it runs.

Everything — the field evolution, the relativistic hydrodynamics, the
diagnostics, and the rendering — happens on the GPU. Simulation state is never
copied to the CPU; the volume renderer samples the same device memory the solver
is writing, in the same frame.

## Quick start

```sh
cargo run --release
```

Then drag to orbit, scroll to zoom, and press `W` to fly into the box.

```sh
# a bigger lattice
cargo run --release -- --grid 256

# one bubble, so you can watch a single wall and its shock in isolation
cargo run --release -- --grid 128 --bubbles 1 --nucleation simultaneous

# a fast, weakly damped wall (detonation-like)
cargo run --release -- --eta 0.02

# start on a particular view (all four stay switchable in the panel)
cargo run --release -- --field temperature

# no window: print diagnostics and exit (benchmarking and validation)
cargo run --release -- --headless 1500 --report-every 100

# render a figure offscreen, no display needed
cargo run --release -- --headless 700 --field kinetic --screenshot frame.png
```

`--help` lists everything.

## Controls

| | |
|---|---|
| drag left | orbit |
| drag right / middle | pan |
| scroll | zoom |
| `W` / `S` | fly forward / back |
| `space` | run / pause |
| `.` | single step |
| `N` | nucleate a bubble now |
| `R` | reset · `F` reset view · `H` hide panel · `Esc` quit |

## What you are looking at

The default view combines two things:

- **Shaded surfaces** are the bubble walls — the isosurface `phi = phi_b/2` of
  the order parameter. They are tinted by how hard each patch of wall is driving
  the plasma.
- **The glow** is the plasma. Four fields are selectable in the panel; the
  interesting ones are *temperature contrast*, where the compression shell ahead
  of a slow wall and the reheated bubble interior stand out, and *kinetic
  energy*, which is what survives after the walls are gone.

Use the **cutaway** slider to slice the box open and look inside, and drop
**wall opacity** to see the fluid structure through the walls.

Things worth doing:

1. Start with `--bubbles 1 --nucleation simultaneous` and watch a single wall
   accelerate to its terminal speed. The panel shows the force-balance estimate
   to compare against.
2. Drag the **friction** slider while it runs. High friction gives a slow
   deflagration with a shock running *ahead* of the wall; low friction gives a
   fast detonation with everything behind it.
3. Let the default 12-bubble run complete, then switch the volume field to
   *kinetic energy*. What is left is the acoustic field — this is the part that
   makes gravitational waves.
4. Watch **energy drift**. Total energy is conserved by the continuum equations,
   so that number is a direct readout of discretisation error. Push the Courant
   number up and watch it degrade.

## The model in one paragraph

A real scalar order parameter `phi` with a quartic potential that has a barrier
(hence *first order*), coupled to a relativistic perfect fluid with the **bag
equation of state** (`e = 3p`, `c_s = 1/sqrt(3)`) through a phenomenological
friction term `eta`. The friction is the only modelling choice: the fluid's
energy-momentum source follows from it by conservation of the total stress
tensor, so the two sectors exchange energy exactly and the total is conserved by
construction. Bubbles are seeded explicitly above the critical radius, as in
published simulations of this system — thermal nucleation is far too rare to
resolve on a lattice.

Full details, including an explicit list of what the model leaves out, are in
[docs/PHYSICS.md](docs/PHYSICS.md).

## Numerics

- **Field:** second-order central differences, no artificial dissipation.
- **Fluid:** second-order MUSCL reconstruction of the primitive variables with a
  monotonised-central limiter and an HLL approximate Riemann solver — shock
  capturing matters, because the shocks and the sound waves *are* the physics.
- **Time:** strong-stability-preserving RK3, chosen over RK2 because the field
  sector has purely imaginary eigenvalues that RK2 would slowly amplify.

Measured on an RTX 3090: **1.25 G cell-updates/s** sustained, a uniform state
preserved to machine precision, and total energy conserved to `2.6e-5` over 1500
steps. [docs/NUMERICS.md](docs/NUMERICS.md) has the numbers and the stability
analysis.

## Extending it

The two things most likely to need replacing are isolated behind small, explicit
interfaces:

- **Equation of state** — `shaders/eos.wgsl`. The solver calls ten functions and
  assumes nothing else. [docs/EOS.md](docs/EOS.md) works through three
  extensions: phase-dependent degrees of freedom, a non-conformal equation of
  state needing an iterative inversion, and a temperature-dependent effective
  potential (which restores a real limitation of the current model).
- **Scalar potential** — `shaders/potential.wgsl`.

**Precision.** The solver is single precision, and every physical quantity in
the shaders is declared `flt` rather than `f32` so the width is a single-point
decision. Converting to double is a *backend* change, not a flag: WGSL has no
`f64` type at all, and on GA102 hardware FP64 runs at 1/64 the FP32 rate.
[docs/PRECISION.md](docs/PRECISION.md) covers what was done to make `f32` behave
well (the primitive-recovery inversion is rewritten to avoid catastrophic
cancellation), which regimes genuinely need more, and four migration paths
ordered by benefit-per-effort — the cheapest two need no `f64` at all.

## Layout

```
shaders/
  common.wgsl      parameters, lattice indexing, slope limiters
  potential.wgsl   scalar potential            <- swap point
  eos.wgsl         equation of state           <- swap point
  primitives.wgsl  conserved -> primitive inversion
  step.wgsl        one SSP-RK3 stage: fluxes, sources, combination
  nucleate.wgsl    stamp in super-critical bubbles
  init.wgsl        homogeneous symmetric phase at rest
  vis.wgsl         pack state into the 3D texture the renderer samples
  reduce.wgsl      global diagnostics, on-GPU
  volume.wgsl      ray-marching volume renderer
src/
  physics.rs       parameter derivation, nucleation schedule
  capture.rs       offscreen render to PNG
  gpu/sim.rs       buffers, pipelines, the time step
  gpu/render.rs    volume renderer
  gpu/preprocess.rs  WGSL include resolution + precision alias injection
  headless.rs      windowless runs for benchmarking and validation
  app.rs           window, device, frame loop
```

## Requirements

A GPU with Vulkan, Metal, or DX12. Developed against an **NVIDIA RTX 3090**;
the defaults are sized for it. Memory is roughly 88 bytes per cell, so 192³
needs 0.62 GB and 256³ needs 1.48 GB.

`cargo test` runs 18 tests covering the parameter derivation, shader assembly,
buffer rotation, and struct layouts — no GPU required. `--headless` exercises
the solver against a GPU without a display, and `--screenshot` does the same
for the renderer.

## License

MIT
