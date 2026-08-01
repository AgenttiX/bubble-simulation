# Replacing the equation of state

The hydrodynamic solver never assumes a particular equation of state. It calls
the functions in `shaders/eos.wgsl` and nothing else. Replace their bodies and
the rest of the code is untouched.

The same applies to the shape of the free energy in the field direction, which
lives entirely in `shaders/potential.wgsl`.

## The interface

| function | contract |
|---|---|
| `eos_sound_speed2(p, phi)` | `c_s^2 = dp/de` at fixed entropy |
| `eos_enthalpy_from_pressure(p, phi)` | `w = e + p` |
| `eos_energy_from_pressure(p, phi)` | `e(p)` |
| `eos_temperature(p, phi)` | `T(p)`, used only for visualisation |
| `eos_pressure_from_conserved(E, Z2, phi)` | invert `(E, \|Z\|^2) -> p` |
| `eos_prim_from_cons(U, phi)` | full inversion to `(v, p)` |
| `eos_cons_from_prim(q, phi)` | `(v, p) -> (E, Z)` |
| `eos_flux(q, U, axis)` | physical flux along `axis` |
| `eos_wave_speeds(q, phi, axis)` | `(lambda_min, lambda_max)` |
| `eos_regularise(U)` | clamp a conserved state into the physical domain |

Every one already takes `phi`, even though the bag model ignores it. That is
deliberate: it is the hook for a phase-dependent equation of state, and it means
adding one does not change a single call site.

The rest of the solver depends on exactly two structural facts, which any
replacement must preserve:

- `E = w W^2 - p` and `Z^i = w W^2 v^i`, hence `Z^i = (E + p) v^i`;
- the wave speeds bound the true characteristics (HLL needs an *outer* estimate,
  so erring wide costs accuracy, never stability).

## Why the default is what it is

The bag model with `e = 3p` gives a closed-form inversion. That is the only
reason it is special. The generalisation is straightforward: writing
`s = E + p = w W^2`,

```
v   = |Z| / s
W^2 = s^2 / (s^2 - |Z|^2)
w   = s / W^2 = (s^2 - |Z|^2) / s
e   = w - p
```

so closing with `p = p_EoS(e)` leaves one scalar equation in one unknown:

```
f(p) = p - p_EoS( ((E+p)^2 - Z^2)/(E+p) - p ) = 0
```

which is monotone in `p` over the physical range and converges in a handful of
Newton or bisection steps from the previous timestep's value.

## Worked example 1: different degrees of freedom in the two phases

The most common upgrade. In the real bag model the number of effective
relativistic degrees of freedom differs across the wall:

```
p_symmetric = (1/3) a_s T^4 - eps ,   p_broken = (1/3) a_b T^4
```

with `a_s > a_b`, since some species become massive in the broken phase. The
default code has `a_s = a_b = a`.

To add it, interpolate `a` across the wall with the order parameter:

```wgsl
fn eos_dof(phi: flt) -> flt {
    // Smoothly interpolate between a_s at phi = 0 and a_b at phi = phi_b.
    let h = clamp(phi / P.phi_b, flt(0.0), flt(1.0));
    let smooth = h * h * (flt(3.0) - flt(2.0) * h);
    return P.a_rad * (flt(1.0) + (P.dof_ratio - flt(1.0)) * smooth);
}

fn eos_temperature(p: flt, phi: flt) -> flt {
    return sqrt(sqrt(flt(3.0) * max(p, P.p_floor) / eos_dof(phi)));
}
```

`e = 3p` and `c_s^2 = 1/3` still hold on both sides -- only the relation between
`p` and `T` changes -- so the inversion is unchanged and only `eos_temperature`
needs editing. Add `dof_ratio` to `SimParams` in `shaders/common.wgsl` and to
`SimParamsGpu` in `src/gpu/sim.rs`, keeping the two layouts identical.

## Worked example 2: a genuinely non-conformal equation of state

Say `e = p / (Gamma - 1) + rho`, or a tabulated `c_s^2(T)` from lattice QCD.
Now the inversion needs iterating. Replace `eos_pressure_from_conserved`:

```wgsl
fn eos_pressure_from_conserved(e_lab: flt, z2: flt, phi: flt) -> flt {
    // Bracket: p must be positive and below the ultra-relativistic bound.
    var p = max(flt(1.0) / flt(3.0) * e_lab, P.p_floor);
    for (var it = 0; it < 8; it = it + 1) {
        let s = e_lab + p;
        let e = (s * s - z2) / s - p;
        let residual = p - eos_pressure_of_energy(e, phi);
        // dp_EoS/de = c_s^2, and de/dp = -(1 + z2/s^2) - 1 for this closure.
        let de_dp = -(flt(1.0) + z2 / (s * s)) - flt(1.0);
        let deriv = flt(1.0) - eos_sound_speed2(p, phi) * de_dp;
        p = max(p - residual / deriv, P.p_floor);
    }
    return p;
}
```

Then also update `eos_sound_speed2` (it is no longer constant, so the wave-speed
formula picks up the change for free) and `eos_enthalpy_from_pressure`.

Two things to watch:

- **Cost.** The inversion runs once per cell per RK stage, so eight Newton
  iterations is roughly a 2-3× cost on the `primitives` pass. That pass is a
  small share of the frame, so the effect on total throughput is modest, but
  measure it with `--headless`.
- **Convergence near `|v| -> 1`.** The residual flattens; keep the floor on `p`
  and cap the iteration count rather than looping to a tolerance, so a single
  bad cell cannot stall a whole workgroup.

## Worked example 3: a temperature-dependent potential

This is the physically most significant extension, because it restores something
the current model genuinely lacks: **the wall does not feel the reheating of the
plasma in front of it** (see [PHYSICS.md](PHYSICS.md#what-is-left-out)). With
`V = V(phi)` only, the driving pressure `eps` is fixed, and a deflagration never
slows down as the fluid ahead heats up.

To restore it, move to the finite-temperature effective potential

```
V(phi, T) = 1/2 gamma (T^2 - T_0^2) phi^2 - 1/3 A T phi^3 + 1/4 lambda phi^4
```

The changes are:

1. `shaders/potential.wgsl`: give `potential` and `dpotential` a second
   argument `T`.
2. `shaders/step.wgsl`: the local temperature is already available -- the
   primitive pressure is in `prim[i0]`, so `eos_temperature(q0.w, phi0)` is one
   call. Pass it to `dpotential`.
3. `shaders/eos.wgsl`: the thermodynamics is no longer that of the fluid alone.
   With `p_tot(phi,T) = (1/3) a T^4 - V(phi,T)`, entropy is `s = dp_tot/dT` and
   energy is `e = T s - p_tot`, so `eos_energy_from_pressure` picks up the
   `-T dV/dT` term and the inversion becomes the iterative one from example 2.
4. `shaders/reduce.wgsl`: the diagnostics' `potential(phi)` term needs the same
   second argument, or the energy-conservation readout will be wrong.

Step 4 is the one that is easy to miss. If the drift readout suddenly looks bad
after an equation-of-state change, check that the diagnostic is measuring the
same energy the solver is conserving.

## Checklist for any change

1. Edit `shaders/eos.wgsl` and/or `shaders/potential.wgsl`.
2. If new parameters are needed, add them to `SimParams` in
   `shaders/common.wgsl` **and** `SimParamsGpu` in `src/gpu/sim.rs`. The two
   layouts must match byte for byte; `parameter_blocks_have_the_expected_size`
   guards the size, not the field order.
3. Update `reduce.wgsl` if the total energy expression changed.
4. Run `cargo test` — the shader assembly tests catch syntax errors without a
   GPU.
5. Run `--headless` with one bubble and check that the quiescent state stays
   quiescent and that the energy drift is still small. Those two checks catch
   most mistakes.
