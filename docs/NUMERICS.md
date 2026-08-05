# Discretisation, stability, and what was measured

## Lattice

Cubic, periodic, spacing `dx = 1` by choice of length units. Linear index
`i = (z ny + y) nx + x`, so consecutive threads in a workgroup (which vary in
`x`) touch consecutive addresses.

State is split across two buffers so both stay naturally aligned:

| buffer | contents | bytes/cell |
|---|---|---|
| `field` | `(phi, pi)` | 8 |
| `fluid` | `(E, Zx, Zy, Zz)` | 16 |
| `prim` | `(vx, vy, vz, p)`, recomputed each stage | 16 |

Three copies of `(field, fluid)` are held, the minimum SSP-RK3 needs. With the
`rgba16float` visualisation texture that is **96 bytes per cell**: 0.68 GB at
192³, 1.61 GB at 256³, 4.98 GB at 384³, 12.8 GB at 511³.

Two independent ceilings bound the lattice size, and which one binds depends on
the device:

- **`max_storage_buffer_binding_size`** caps the `fluid` buffer at 16 bytes per
  cell. On an RTX 3090 that limit is 2.147 GB, giving 511³ — and no amount of
  free memory relaxes it.
- **Free device memory** caps the total at 96 bytes per cell. On a 24 GB card
  with a desktop running, that is around 19 GB of budget, or roughly 585³.

So on this hardware the API limit binds first. The Lattice panel reports both
and says which one is doing the capping. Free memory is read through
`VK_EXT_memory_budget`, which reports what the driver will give *this process*
rather than raw free VRAM; measured against `nvidia-smi` it reads about 0.5 GB
more conservative, which is the right direction to err.

## Field sector

Second-order central differences: the 7-point Laplacian and a centred gradient.
No artificial dissipation is added; the scheme's stability comes from the time
integrator, and the field sector is otherwise non-dissipative, which is what
makes the energy-conservation check meaningful.

## Fluid sector

Second-order MUSCL reconstruction of the **primitive** variables `(v, p)` with a
monotonised-central limiter, and an **HLL** approximate Riemann solver at each
cell face. Shock capturing is not optional here: the shock ahead of a
deflagration and the acoustic field left behind are the physics of interest, and
a plain centred scheme would either smear them away or ring.

The relativistic signal speeds along direction `d` are

```
lambda_pm = [ v_d (1 - c_s^2)
              +- c_s sqrt( (1-v^2) [ 1 - v_d^2 - c_s^2 (v^2 - v_d^2) ] ) ]
            / (1 - v^2 c_s^2)
```

reducing to `+-c_s` at rest and to `+-1` as `|v| -> 1`. Using `min(.,0)` and
`max(.,0)` on the HLL wave speeds makes the flux degrade gracefully to pure
upwinding on supersonic faces.

Reconstruction happens on primitives rather than conserved variables, and every
reconstructed face state is passed through a `sanitise` step that floors the
pressure and rescales any super-luminal velocity. Reconstruction can otherwise
produce states outside the physical domain near a strong shock.

### Primitive recovery

Given `E` and `|Z|`, with `s = E + p = w W^2`, the closure for a general
equation of state is one scalar equation in `p`:

```
f(p) = p - p_EoS( ((E+p)^2 - Z^2)/(E+p) - p ) = 0
```

For `e = 3p` this is the quadratic `3p^2 + 2 E p - (E^2 - Z^2) = 0`, whose
physical root is `p = ( sqrt(4E^2 - 3Z^2) - E ) / 3`.

**That form is not what the code uses.** When `Z << E` -- the slow-flow regime
that most of the box is in, most of the time -- the square root approaches `2E`
and the subtraction loses precision catastrophically. Multiplying by the
conjugate gives the algebraically identical but numerically stable

```
p = (E^2 - Z^2) / ( E + sqrt(4E^2 - 3Z^2) )
```

which is worth roughly two decimal digits in single precision. This is the
single most important precision decision in the solver; see
[PRECISION.md](PRECISION.md).

Conserved states are clamped to `|Z| <= 0.9999 E` before inversion, which is
exactly the condition for `|v| < 1` and `p > 0`. The clamp is applied to the
*local copy* used for inversion, not written back, so it does not silently
corrupt the conserved variables.

## Time integration

Strong-stability-preserving RK3 (Shu-Osher). Every stage has the same shape,

```
U_out = a0 U^n + a1 ( U_in + dt L(U_in) )
```

with `(a0, a1)` = `(0, 1)`, `(3/4, 1/4)`, `(1/3, 2/3)`, so one shader and one
uniform per stage cover all three. Each stage is a convex combination, which is
what makes the scheme SSP.

RK3 rather than RK2 for a specific reason: the field sector is a wave equation
with centred differences, whose semi-discrete eigenvalues are **purely
imaginary**. Heun's method (SSP-RK2) has a stability region that touches the
imaginary axis only at the origin, so it is weakly unstable there and would
grow over the tens of thousands of steps a long run takes. SSP-RK3 includes the
imaginary interval `|lambda dt| <= sqrt(3)`.

### Buffer rotation

With `a = cur`, `b = (cur+1)%3`, `c = (cur+2)%3`:

| stage | reads `U^n` | reads `U^k` | writes |
|---|---|---|---|
| 0 | a | a | b |
| 1 | a | b | c |
| 2 | a | c | b |

The result lands in `b`, which becomes the new `cur`. No stage writes a slot it
reads. A unit test (`stage_slots_never_alias_output_with_input`) asserts this.

### Stability limit

Two constraints, and the field's is the binding one:

- **Field.** The semi-discrete operator has `|lambda|_max = 2 sqrt(3) / dx`, and
  SSP-RK3 permits `|lambda| dt <= sqrt(3)`, giving `dt <= 0.5 dx`.
- **Fluid.** Unsplit 3D MUSCL-HLL needs `dt <= CFL dx / sum_d |lambda_d|`. In
  quiescent plasma each direction contributes `c_s = 0.577`, giving
  `dt <= 0.58 dx`; as `|v| -> 1` all three approach 1 and the bound tightens
  toward `dt <= 0.33 dx`.

So `cfl = 0.5` is the hard ceiling and the code rejects anything above it.
**Accuracy degrades long before stability does**, which is measured below.

## Diagnostics

Two dispatches: 1024 workgroups of 256 threads grid-striding the lattice into
partial sums (plus one atomic max per workgroup for `max |v|`), then a single
workgroup folding the partials. Result is 32 bytes.

`max |v|` uses `atomicMax` on the bit pattern of a non-negative `f32`, which is
valid because IEEE-754 ordering of non-negative floats matches unsigned integer
ordering of their bit patterns.

Those 32 bytes are the only thing that ever crosses back to the host, and even
that is a non-blocking mapped read consumed a frame or two later. If the
previous readback has not been consumed, the reduction is simply skipped that
frame.

## Measured behaviour

All runs via `--headless`, single precision, on an RTX 3090.

### Quiescent state is exactly preserved

64³, no bubble nucleated within the run:

```
step   broken   max|v|            K
 100    0.00%   0.0000   8.373e-14
 400    0.00%   0.0000   1.340e-12
```

`max |v|` is exactly zero and the kinetic energy stays at roundoff. The scheme
is well balanced: a uniform state is a discrete fixed point, so any motion seen
later is physics and not scheme noise.

### Single bubble, default parameters

128³, one bubble seeded at `t = 0`, `eta = 0.3`:

```
step   time   broken    R_eq   max|v|          K      E/E0
 100   20.0    3.10%    24.9   0.0313   4.587e-5  1.000000
 300   60.0    9.68%    36.5   0.0549   4.034e-4  0.999988
 500  100.0   26.25%    50.8   0.0790   1.569e-3  0.999967
 700  140.0   60.13%    67.0   0.1418   4.393e-3  0.999954
1000  200.0   99.29%    79.2   0.3333   1.842e-2  1.000001
1500  300.0  100.00%    79.4   0.3569   1.229e-2  1.000026
```

The wall accelerates from `dR/dt ~ 0.27` to a terminal `~0.40`, against the
`0.45` the force-balance estimate predicts when the fluid back-reaction is
neglected -- the right ordering. Energy is conserved to `2.6e-5` over 1500
steps. Note that part of even *that* is the reduction itself: summing 2 million
`f32` values accumulates a relative error of order `sqrt(N) * eps ~ 1e-4`, so
the solver's own drift is bounded above by what is shown here, not measured by
it.

The final `K ~ 1.2e-2` at `alpha = 0.1` is in the range expected from the
standard efficiency factors for a deflagration at `v_w ~ 0.4`, which is a
non-trivial check that the field-fluid coupling has the right magnitude and
sign.

### Friction controls the regime

Same setup, `eta = 0.02`: `R_eq` goes 29.9 → 45.7 → 63.3 over equal intervals,
a wall speed near 0.85 rather than 0.4. Energy drift grows to `-4.5e-3`, and the
reason is physical rather than a bug: a wall at `gamma ~ 2` is Lorentz
contracted to `l_w / gamma`, under two cells, and the lattice can no longer
resolve it. **A runaway wall is always under-resolved.** Watch `max |v|` when
lowering `eta`.

### Courant number

96³, one bubble, `eta = 0.02`, integrated to the same physical time `t = 450`:

| `cfl` | steps | energy drift |
|---|---|---|
| 0.2 | 2250 | `-8.2e-4` |
| 0.5 | 900 | `-1.06e-2` |

Both are stable; 0.5 is 13× less accurate. Hence the default of 0.2.

### Throughput

Sustained **1.25 G cell-updates/s** across lattice sizes from 64³ to 256³ — that
is three RK stages, each with a primitive-recovery pass and a flux pass, so
roughly 7.5 G cell-passes/s. Concretely: 585 steps/s at 128³, 177 steps/s at
192³. With `steps_per_frame = 2` the default 192³ lattice runs comfortably above
60 fps with the renderer active.

## Reproducing these numbers

Every table above comes from `--headless`, which needs a GPU but no display:

```sh
cargo run --release -- --grid 128 --bubbles 1 --nucleation simultaneous \
    --headless 1500 --report-every 100
```

`--screenshot PATH` renders one frame offscreen after the run, which is how the
renderer is checked on a headless machine.

## Known limitations of the scheme

- The clamp on `|Z|` violates conservation in cells where it fires. It should
  only fire in pathological states; if the energy drift is large, that is the
  first thing to suspect.
- Nucleation is applied on a frame boundary, so an event's time is quantised to
  `steps_per_frame * dt`. Irrelevant against a nucleation duration of tens of
  time units, but it means the schedule is not reproducible across different
  `steps_per_frame`.
- Changing physics parameters mid-run changes the energy of the current
  configuration. The UI says so and re-zeroes the baseline.
- The reconstruction stencil reaches two cells, so lattice dimensions below 8
  are rejected.
