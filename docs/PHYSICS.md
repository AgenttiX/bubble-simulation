# The physical model

Units are natural throughout: `c = hbar = k_B = 1`, the metric signature is
`(+,-,-,-)`, and space is flat (see [Expansion](#what-is-left-out) for why).

## The picture

Early in its history the universe was hot enough to sit in a symmetric phase of
some field theory. As it cooled, a second, lower-energy phase became available,
but a barrier in the free energy separated the two. That makes the transition
*first order*: it does not proceed smoothly everywhere at once. Instead, rare
thermal or quantum fluctuations nucleate bubbles of the new phase, and those
bubbles grow, sweep up the surrounding plasma, and collide.

Three things happen that this simulation is built to show:

1. **Walls accelerate and then reach a terminal speed.** The vacuum energy
   released per unit volume pushes the wall outwards; the plasma pushes back.
2. **The plasma is stirred.** A moving wall drags the fluid, producing a shell
   of bulk motion around each bubble, and compressing the fluid ahead of it.
3. **Sound waves survive the transition.** Once all bubbles have merged and the
   walls are gone, the box is left ringing. This long-lived acoustic field is
   the dominant source of gravitational waves from a cosmological phase
   transition, which is why the kinetic energy fraction `K` is displayed live.

## Degrees of freedom

A real scalar order parameter `phi` coupled to a relativistic perfect fluid.

**Field.** Stress tensor

```
T^{mu nu}_phi = d^mu phi d^nu phi - g^{mu nu} ( 1/2 d_lambda phi d^lambda phi - V(phi) )
```

**Fluid.** Enthalpy `w = e + p`, four-velocity `u^mu = W(1, v)` with
`W = 1/sqrt(1-v^2)`:

```
T^{mu nu}_fluid = w u^mu u^nu - g^{mu nu} p
```

## The potential

```
V(phi) = 1/2 m2 phi^2  -  1/3 delta phi^3  +  1/4 lambda phi^4
```

The three coefficients are not independent inputs. They are solved for from the
three conditions that make this a first-order transition with a prescribed
latent heat, given `lambda`, `phi_b` and `eps`:

| condition | meaning |
|---|---|
| `V(0) = 0`, `V'(0) = 0`, `V''(0) = m2 > 0` | the symmetric phase is metastable |
| `V'(phi_b) = 0` | the broken phase is stationary |
| `V(phi_b) = -eps` | the vacuum energy released is `eps` |

which give

```
m2    = lambda phi_b^2 / 2  -  6 eps / phi_b^2
delta = 3 lambda phi_b / 2  -  6 eps / phi_b^3
```

and require `eps < lambda phi_b^4 / 12` for the barrier to exist at all. The
code parameterises that constraint away with `eps_ratio = 12 eps / (lambda
phi_b^4)` in `(0,1)`; at `eps_ratio -> 0` the well is degenerate and
`V = (lambda/4) phi^2 (phi - phi_b)^2`, at `eps_ratio -> 1` the barrier
disappears. `src/physics.rs` has a unit test asserting all three conditions hold
for the derived coefficients.

Useful quantities that follow, all shown live in the panel:

| quantity | expression | default |
|---|---|---|
| wall thickness | `l_w = 2 sqrt(2) / (phi_b sqrt(lambda))` | 4.0 cells |
| surface tension | `sigma = int sqrt(2V) dphi = sqrt(lambda/2) phi_b^3 / 6` | 0.083 |
| critical radius | `R_c = 2 sigma / eps` | 13.3 cells |

`l_w` and `R_c` are the two numbers that decide whether a given parameter choice
is resolvable on the lattice: you want `l_w` above ~3 cells, `R_c` comfortably
larger than `l_w`, and the box much larger than the final bubble size.

## Coupling: friction

The field equation carries a phenomenological friction term of strength `eta`,
which stands in for the microphysics of particles scattering off the moving
wall:

```
d_mu d^mu phi  +  dV/dphi  =  - eta u^mu d_mu phi                      (1)
```

The fluid's share follows without any further modelling choice. For *any*
scalar field,

```
d_mu T^{mu nu}_phi = ( box phi + V'(phi) ) d^nu phi
```

so substituting (1) gives `d_mu T^{mu nu}_phi = - eta (u^lambda d_lambda phi)
d^nu phi`, and since the total stress tensor is conserved the fluid must gain
exactly that:

```
d_mu T^{mu nu}_fluid = + eta ( u^lambda d_lambda phi ) d^nu phi        (2)
```

The friction terms therefore cancel pairwise between the two sectors, and total
energy is conserved by construction. That is what the **energy drift** readout
checks, and it is a genuine test: it catches sign errors in the coupling, not
just arithmetic.

## Evolved system

Writing `pi = dphi/dt` and using the lab-frame fluid densities
`E = T^{00}_fluid` and `Z^i = T^{0i}_fluid`, the six evolved fields obey

```
dphi/dt = pi
dpi /dt = laplacian(phi) - dV/dphi          - eta D
dE  /dt = -div(Z)                           + eta D pi
dZ^i/dt = -d_j ( Z^i v^j + p delta^{ij} )   - eta D d_i phi
```

with `D = u^mu d_mu phi = W (pi + v . grad phi)`. The identity `E + p = w W^2`,
hence `Z^i = (E + p) v^i`, is what makes the flux for `E` cheap to write.

Boundary conditions are periodic in all three directions.

## Equation of state

The default is the **bag model**: the plasma is an ideal gas of massless
species,

```
p_fluid = (1/3) a T^4 ,   e_fluid = 3 p_fluid ,   c_s^2 = 1/3
```

and the phase-dependent vacuum energy is carried by `V(phi)`, so that the total
pressure is `p_fluid - V(phi)`. The bag constant is `eps`, and the transition
strength is

```
alpha = eps / e_ref
```

the latent heat relative to the plasma's radiation energy density. This is the
standard measure of how strong a transition is; `alpha = 0.1` is a strong but
not extreme transition.

See [EOS.md](EOS.md) for the six functions that define this and how to replace
them.

## Wall velocity

The wall reaches terminal velocity when the driving pressure balances friction.
Integrating the friction force across the wall gives roughly

```
eps  ~  eta sigma gamma_w v_w
```

which is what the panel reports as the estimated wall speed. It is an
*upper* bound in practice, because it neglects the back-reaction of the fluid:
as the wall pushes the plasma in front of it, the relative velocity in the wall
frame drops and the friction term weakens.

The two limits are worth exploring interactively:

- **Large `eta`** (slow wall, subsonic): a deflagration. The fluid is pushed
  outwards ahead of the wall, forming a compression shell and eventually a
  shock that runs ahead of the wall at a speed above `c_s`. Switch the volume
  field to *temperature contrast* to see it clearly.
- **Small `eta`** (fast wall, supersonic): a detonation. The fluid ahead is
  undisturbed until the wall arrives, and the bulk motion is confined to a
  rarefaction tail behind the wall.

## Nucleation

Thermal nucleation is exponentially rare on the scale of a lattice cell, so no
lattice simulation of this kind resolves it, and this one does not pretend to.
Bubbles are seeded explicitly, as in the published simulations of this system,
with the planar-kink profile wrapped onto a sphere:

```
phi(r) = (phi_b / 2) [ 1 - tanh( (r - r_0) / l_w ) ]
```

at `r_0 = 1.6 R_c`, comfortably super-critical so the bubble grows rather than
collapsing. Two schedules are offered:

- **simultaneous** — every bubble at `t = 0`. All bubbles are the same size at
  any moment, which makes the geometry of the collision network easy to read.
- **exponential** — nucleation rate proportional to `exp(beta t)`, the
  physically relevant case, which gives a broad distribution of bubble sizes
  because early bubbles have grown large by the time the last ones appear.

Stamping a bubble in by hand adds its surface and vacuum energy to the box, so
the energy-conservation readout steps at each nucleation event. The baseline
re-zeroes automatically once the last bubble is in.

## What is left out

Being explicit about this matters more than the list being short.

- **Cosmological expansion.** The box is Minkowski. For transitions where the
  bubble lifetime is short compared to a Hubble time -- the usual case -- this
  is a good approximation. Adding it means moving to conformal coordinates and
  inserting Hubble friction terms.
- **Temperature dependence of the potential.** `V` depends on `phi` only. The
  consequence is physical and worth knowing: the wall does **not** feel the
  reheating of the plasma in front of it, because the driving pressure `eps` is
  fixed. Only the friction term responds to the fluid. Real deflagrations slow
  down as the plasma ahead heats up and the free-energy difference shrinks.
  [EOS.md](EOS.md) shows how to restore this.
- **Gravitational waves.** Not computed. The kinetic energy fraction `K` shown
  in the panel is the quantity that sources them, so it is the right first
  diagnostic, but extracting a spectrum needs a transverse-traceless projection
  of the stress tensor accumulated over time.
- **Any back-reaction on the metric**, and any second field.
