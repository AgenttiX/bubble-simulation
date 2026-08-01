// ============================================================================
//  eos.wgsl -- equation of state of the plasma.   *** SWAP POINT #2 ***
// ============================================================================
//
//  The hydrodynamic solver never assumes a particular equation of state
//  directly.  It only calls the six functions below.  Replace their bodies and
//  the rest of the code is untouched.  docs/EOS.md walks through a worked
//  example.
//
//      eos_sound_speed2(p, phi)          -> c_s^2
//      eos_enthalpy_from_pressure(p,phi) -> w = e + p
//      eos_temperature(p, phi)           -> T
//      eos_pressure_from_conserved(E, Z2, phi) -> p
//      eos_cons_from_prim(v, p, phi)     -> (E, Zx, Zy, Zz)
//      eos_flux(v, p, U, axis)           -> flux of (E, Z) along `axis`
//
//  DEFAULT: THE BAG MODEL
//  ----------------------
//  The plasma is an ideal relativistic gas of massless species,
//
//      p_fluid = (1/3) a T^4 ,     e_fluid = a T^4 = 3 p_fluid ,   c_s^2 = 1/3
//
//  and the vacuum energy of the bag -- the constant that differs between the
//  two phases -- is carried by the scalar potential V(phi) in potential.wgsl.
//  Total pressure and energy density are
//
//      p_tot = p_fluid - V(phi)
//      e_tot = e_fluid + 1/2 pi^2 + 1/2 |grad phi|^2 + V(phi)
//
//  which is exactly the bag model with bag constant eps = V(0) - V(phi_b),
//  split so that the field sector owns the phase-dependent part.  Because the
//  radiation constant `a` is the same in both phases here, the number of
//  effective relativistic degrees of freedom does not jump across the wall;
//  see docs/EOS.md for how to make `a` depend on phi.
//
//  CONSERVATIVE VARIABLES
//  ----------------------
//  With metric signature (+,-,-,-), u^mu = W(1, v), W = 1/sqrt(1-v^2) and
//  T^{mu nu}_fluid = w u^mu u^nu - g^{mu nu} p, the lab-frame densities are
//
//      E    = T^{00} = w W^2 - p
//      Z^i  = T^{0i} = w W^2 v^i
//
//  and note the useful identity  E + p = w W^2, hence  Z^i = (E + p) v^i.

// ---------------------------------------------------------------------------
//  Thermodynamics
// ---------------------------------------------------------------------------

fn eos_sound_speed2(p: flt, phi: flt) -> flt {
    // Ultra-relativistic ideal gas.  For a general EoS this is dp/de at fixed
    // entropy and will depend on p (and possibly on the phase, via phi).
    return flt(1.0) / flt(3.0);
}

fn eos_enthalpy_from_pressure(p: flt, phi: flt) -> flt {
    // w = e + p = 4p for e = 3p.
    return flt(4.0) * p;
}

fn eos_energy_from_pressure(p: flt, phi: flt) -> flt {
    return flt(3.0) * p;
}

fn eos_temperature(p: flt, phi: flt) -> flt {
    // p = (1/3) a T^4  =>  T = (3 p / a)^(1/4)
    let x = flt(3.0) * max(p, P.p_floor) / P.a_rad;
    return sqrt(sqrt(x));
}

// ---------------------------------------------------------------------------
//  Primitive recovery:  (E, |Z|^2)  ->  p
// ---------------------------------------------------------------------------
//
//  General procedure for *any* EoS.  Write s = E + p = w W^2.  Then
//      v   = |Z| / s
//      W^2 = s^2 / (s^2 - |Z|^2)
//      w   = s / W^2 = (s^2 - |Z|^2) / s
//      e   = w - p
//  and closing with the EoS gives one scalar equation for p:
//      f(p) = p - p_EoS( ((E+p)^2 - Z^2)/(E+p) - p ) = 0
//  which is solved by a few Newton or bisection iterations (see docs/EOS.md).
//
//  For e = 3p the equation is a quadratic,  3p^2 + 2 E p - (E^2 - Z^2) = 0,
//  with the physical root
//      p = ( sqrt(4E^2 - 3Z^2) - E ) / 3 .
//  That form suffers catastrophic cancellation when Z << E (the non-relativistic
//  limit, where the square root approaches 2E).  Multiplying by the conjugate
//  gives the algebraically identical but numerically stable
//      p = (E^2 - Z^2) / ( E + sqrt(4E^2 - 3Z^2) )
//  which is what we use.  In single precision this is worth roughly two decimal
//  digits in the slow-flow regions that make up most of the box.

fn eos_pressure_from_conserved(e_lab: flt, z2: flt, phi: flt) -> flt {
    let disc = max(flt(4.0) * e_lab * e_lab - flt(3.0) * z2, flt(0.0));
    let p = (e_lab * e_lab - z2) / (e_lab + sqrt(disc));
    return max(p, P.p_floor);
}

// Clamp a conserved state into the physically admissible region.  |Z| < E is
// required for |v| < 1; numerical error in shocks can push a cell just outside.
fn eos_regularise(u: vec4<flt>) -> vec4<flt> {
    let e_lab = max(u.x, P.e_floor);
    let z = u.yzw;
    let zlen = length(z);
    let zcap = P.z_max * e_lab;
    var zc = z;
    if (zlen > zcap) {
        zc = z * (zcap / max(zlen, flt(1e-30)));
    }
    return vec4<flt>(e_lab, zc.x, zc.y, zc.z);
}

// Full conserved -> primitive step.  Returns (vx, vy, vz, p).
fn eos_prim_from_cons(u_raw: vec4<flt>, phi: flt) -> vec4<flt> {
    let u = eos_regularise(u_raw);
    let e_lab = u.x;
    let z = u.yzw;
    let p = eos_pressure_from_conserved(e_lab, dot(z, z), phi);
    let s = e_lab + p;                       // = w W^2
    let v = z / max(s, flt(1e-30));
    return vec4<flt>(v.x, v.y, v.z, p);
}

// ---------------------------------------------------------------------------
//  Primitive -> conservative, and the physical fluxes
// ---------------------------------------------------------------------------

fn eos_cons_from_prim(q: vec4<flt>, phi: flt) -> vec4<flt> {
    let v = q.xyz;
    let p = q.w;
    let v2 = min(dot(v, v), P.z_max * P.z_max);
    let w_gam2 = eos_enthalpy_from_pressure(p, phi) / (flt(1.0) - v2);   // w W^2
    return vec4<flt>(w_gam2 - p, w_gam2 * v.x, w_gam2 * v.y, w_gam2 * v.z);
}

// Flux of (E, Z^x, Z^y, Z^z) in direction `axis`:
//     F_E    = Z^d = (E + p) v^d
//     F_Z^i  = Z^i v^d + p delta^{i d}
fn eos_flux(q: vec4<flt>, u: vec4<flt>, axis: i32) -> vec4<flt> {
    let v = q.xyz;
    let p = q.w;
    let vd = v[axis];
    var f = vec4<flt>(
        (u.x + p) * vd,
        u.y * vd,
        u.z * vd,
        u.w * vd,
    );
    f[axis + 1] = f[axis + 1] + p;
    return f;
}

// ---------------------------------------------------------------------------
//  Characteristic speeds (needed by the approximate Riemann solver)
// ---------------------------------------------------------------------------
//
//      lambda_pm = [ v_d (1 - c_s^2)
//                    +- c_s sqrt( (1-v^2) [ 1 - v_d^2 - c_s^2 (v^2 - v_d^2) ] ) ]
//                  / (1 - v^2 c_s^2)
//
//  which reduces to +-c_s at rest and to +-1 as |v| -> 1.

fn eos_wave_speeds(q: vec4<flt>, phi: flt, axis: i32) -> vec2<flt> {
    let v = q.xyz;
    let v2 = min(dot(v, v), P.z_max * P.z_max);
    let vd = v[axis];
    let cs2 = eos_sound_speed2(q.w, phi);
    let cs = sqrt(cs2);

    let denom = flt(1.0) - v2 * cs2;
    let rad = max((flt(1.0) - v2) * (flt(1.0) - vd * vd - cs2 * (v2 - vd * vd)), flt(0.0));
    let a = vd * (flt(1.0) - cs2) / denom;
    let b = cs * sqrt(rad) / denom;
    return vec2<flt>(a - b, a + b);
}
