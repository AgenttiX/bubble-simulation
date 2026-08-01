# Single precision, and what moving to double would take

The solver is `f32` throughout. This document says why, what was done to make
`f32` behave well, and exactly what a double-precision port would involve.

## The blocker, stated plainly

**WGSL has no 64-bit floating point type.** Not "not yet exposed by wgpu" — the
language has `f32`, `f16` (behind an extension), `i32`, `u32`, and no `f64`.
So double precision is not a flag in this codebase; it is a backend change.

And there is a second, independent problem specific to the hardware target.
The RTX 3090 is a GA102 part, where FP64 runs at **1/64 the FP32 rate** (about
0.56 TFLOPS against 35.6 TFLOPS). Even with a backend that supported `f64`, a
wholesale port on this card would give up roughly two orders of magnitude of
throughput — the simulation would stop being interactive, which is the entire
point of the program. Double precision on consumer Ampere is only sensible for
the parts of the calculation that actually need it.

Neither of these is a reason to write the code as if precision were
unchangeable, which is why the seams described below exist.

## What is already in place

**A single type alias governs the whole solver.** Every physical quantity in the
simulation shaders is declared `flt`, never `f32`:

```wgsl
let disc = max(flt(4.0) * e_lab * e_lab - flt(3.0) * z2, flt(0.0));
```

The alias is injected by `src/gpu/preprocess.rs` ahead of every module, driven
by the `Precision` enum. On the host, `physics::Scalar` plays the same role.
`Precision::Double` exists and panics with a pointer to this file, so the
decision point is visible in the source rather than implied.

**The renderer is precision-independent by construction.** It reads an
`rgba16float` 3D texture and writes an 8-bit framebuffer; nothing there benefits
from more mantissa. So a solver-precision change touches no rendering code at
all.

**The numerically dangerous step was already rewritten.** Primitive recovery
inverts `(E, |Z|) -> p`. The textbook root

```
p = ( sqrt(4E^2 - 3Z^2) - E ) / 3
```

suffers catastrophic cancellation when `Z << E`, which is the slow-flow regime
that most of the box is in most of the time: the square root approaches `2E` and
the subtraction throws away leading digits. The code uses the conjugate form

```
p = (E^2 - Z^2) / ( E + sqrt(4E^2 - 3Z^2) )
```

which is algebraically identical, has no cancellation in that limit, and buys
roughly two decimal digits. **This one change is worth more than a great deal of
what a naive `f64` port would buy**, because it removes the error rather than
padding around it.

**Contrasts are stored, not absolutes.** The visualisation texture holds
`T/T_ref - 1`, not `T`. A fluctuation of relative size `1e-3` is resolved with
full `rgba16float` mantissa instead of landing in the last two bits of a number
near 1.

**Floors are relative.** `p_floor` and `e_floor` are `1e-7 * e_ref`, so they
scale with the chosen parameters instead of being absolute magic numbers that
silently become wrong when `alpha` changes by two orders of magnitude.

## Does `f32` actually suffice here?

Measured, not assumed. Over 1500 steps at 128³ the total energy drifts by
`2.6e-5` relative (see [NUMERICS.md](NUMERICS.md#measured-behaviour)).

The important caveat is that **this number is dominated by the diagnostic, not
the solver.** Naive summation of 2 million `f32` values accumulates a relative
error of order `sqrt(N) * eps ~ 1.4e-4`. So the measurement is an *upper bound*
on the solver's drift, and the solver is doing better than the reported figure.

The practical consequence: if you want a sharper conservation measurement, fix
the reduction before touching the solver. See "Targeted upgrades" below.

## Where `f32` would genuinely bite

1. **Very long runs.** Tens of millions of steps, where per-step roundoff
   random-walks into something visible.
2. **Very weak transitions.** `alpha <~ 1e-4`, where the physical fluid
   perturbation is comparable to `f32` roundoff on the background. This is a
   real regime for gravitational-wave work.
3. **Ultra-relativistic flow.** As `|v| -> 1`, `E^2 - Z^2` cancels; that
   cancellation is physical and no algebraic rearrangement removes it.
4. **Extracting a gravitational-wave spectrum**, where a small transverse-
   traceless residual is accumulated over many steps against a large trace.

If none of these apply, `f32` is the right choice and the 64× throughput is
better spent on lattice resolution.

## Targeted upgrades, cheapest first

These are ordered by benefit-per-unit-effort, and the first two do not need
`f64` at all.

### 1. Compensated summation in the reduction (hours)

Replace the naive accumulation in `shaders/reduce.wgsl` with Kahan-Neumaier
summation, both in the per-thread grid-stride loop and the tree reduction. This
removes the `sqrt(N)` term above and makes the drift readout measure the solver
instead of itself. Cost is negligible: the reduction is a fraction of a percent
of the frame.

### 2. Evolve perturbations rather than totals (days)

Store `E - e_ref` instead of `E`. In the weak-transition regime the whole
difficulty is that a `1e-6` signal rides on an `O(1)` background; subtracting a
known constant background recovers the lost digits exactly, in `f32`, at zero
runtime cost. This is by far the highest-leverage change for regime 2 above, and
it is a better answer than `f64` because it addresses the actual problem.

Requires care in the flux computation, where the background terms must cancel
analytically rather than numerically.

### 3. Double-single (float-float) arithmetic (~1 week)

Represent each value as an unevaluated sum of two `f32`s, giving about 48 bits
of mantissa. Implemented entirely in WGSL with standard two-sum / two-product
algorithms, so it needs **no hardware FP64 and no backend change** — and on
GA102 it is roughly 10-20× the arithmetic cost, still much faster than the
1/64-rate native FP64.

The `flt` alias makes the storage side easy (`alias flt = vec2<f32>`), but every
arithmetic operator must become a function call, so the shader bodies do need
rewriting. Apply it selectively: the primitive recovery and the RK accumulation
are where it pays.

### 4. A backend with native `f64` (~2-4 weeks)

Two routes:

- **Raw Vulkan + GLSL** with the `shaderFloat64` device feature (which the 3090
  supports). Keeps the rendering path, but means hand-writing the Vulkan
  plumbing that wgpu currently provides.
- **CUDA for the compute path** via `cudarc` or `cust`, keeping wgpu for
  rendering and sharing buffers through Vulkan-CUDA external memory interop
  (`VK_KHR_external_memory_fd` / `cudaImportExternalMemory`). More work, but it
  preserves the no-readback property: the CUDA kernels and the Vulkan renderer
  address the same allocation.

Either way, expect roughly 1/64 throughput on this card for the parts actually
running in `f64`, so a mixed-precision split is almost certainly what you want:
`f64` for the conserved-variable accumulation and the inversion, `f32` for
fluxes and the visualisation pass.

## Concrete edit list for a full `f64` port

Assuming a backend that supports it, in dependency order:

1. `src/gpu/preprocess.rs` — make `Precision::Double` emit the alias and the
   backend's enable directive instead of panicking.
2. `shaders/*.wgsl` — no edits needed for the physics shaders; they already
   spell everything `flt`. Two exceptions:
   - `reduce.wgsl` uses `bitcast<u32>` for the atomic max. Either keep that
     accumulator in `f32` (it is a diagnostic; the loss is immaterial) or move
     to a 64-bit atomic.
   - `vis.wgsl` already casts explicitly to `f32` for `textureStore`. Good as is.
3. `src/physics.rs` — `pub type Scalar = f64`, and the `powi`/`sqrt` calls
   follow automatically.
4. `src/gpu/sim.rs` — **the real work.** `SimParamsGpu`, `StageParamsGpu` and
   `BubbleGpu` are `#[repr(C)]` mirrors of WGSL structs. `f64` has 8-byte
   alignment, so the padding changes and the field order may need rearranging to
   avoid holes. The buffer strides (`n_cells * 8` for `field`, `* 16` for
   `fluid`) double. The `parameter_blocks_have_the_expected_size` test will fail
   loudly, which is its purpose.
5. `src/gpu/render.rs` — unchanged. Verify no `Scalar` leaked in.
6. Re-run the validation in [NUMERICS.md](NUMERICS.md#measured-behaviour): the
   quiescent test should now hold to `~1e-30` rather than `~1e-13`, and the
   energy drift should drop by several orders of magnitude.

Step 4 is where the bugs will be, and step 6 is how you find them.
