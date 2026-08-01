// ============================================================================
//  potential.wgsl -- the scalar (order-parameter) potential.
// ============================================================================
//
//  SWAP POINT #1.  Everything the solver knows about the shape of the free
//  energy in the field direction is behind `potential()` and `dpotential()`.
//
//  The default is the standard quartic with a cubic barrier,
//
//      V(phi) = 1/2 m2 phi^2  -  1/3 delta phi^3  +  1/4 lambda phi^4
//
//  which has
//     * a metastable minimum at phi = 0 with V(0) = 0          (symmetric phase)
//     * a barrier                                              (=> first order)
//     * the true minimum at phi = phi_b with V(phi_b) = -eps   (broken phase)
//
//  `m2` and `delta` are *derived* on the host from (lambda, eps, phi_b) so that
//  those three conditions hold exactly:
//
//      m2    = lambda phi_b^2 / 2  -  6 eps / phi_b^2
//      delta = 3 lambda phi_b / 2  -  6 eps / phi_b^3
//
//  (see `src/physics/model.rs`).  For eps = 0 this collapses to the degenerate
//  double well V = (lambda/4) phi^2 (phi - phi_b)^2.
//
//  eps is the bag constant: it is the vacuum energy released per unit volume
//  when a region converts from the symmetric to the broken phase, and it is
//  what drives the bubble wall outwards.
//
//  To use a temperature-dependent effective potential instead -- e.g. the
//  finite-temperature form  V(phi,T) = 1/2 gamma (T^2 - T0^2) phi^2
//  - 1/3 A T phi^3 + 1/4 lambda phi^4  -- give these two functions a second
//  argument `T`, thread the local temperature in from the caller in step.wgsl
//  (it is already available there via `eos_temperature`), and add the
//  corresponding -T dV/dT term to the fluid entropy in eos.wgsl.  See
//  docs/EOS.md for the full recipe.

fn potential(phi: flt) -> flt {
    let p2 = phi * phi;
    return flt(0.5) * P.m2 * p2
         - P.delta * p2 * phi / flt(3.0)
         + flt(0.25) * P.lambda * p2 * p2;
}

fn dpotential(phi: flt) -> flt {
    return phi * (P.m2 - P.delta * phi + P.lambda * phi * phi);
}
