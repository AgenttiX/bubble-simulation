// ============================================================================
//  common.wgsl -- shared parameter block, lattice indexing, small helpers.
// ============================================================================
//
//  PRECISION
//  ---------
//  Every physical quantity in the simulation shaders is declared with the type
//  alias `flt`, never with a literal `f32`.  The alias is injected ahead of
//  this file by the host (see `src/gpu/preprocess.rs`).  Switching the whole
//  solver to another floating point width is therefore a one-line change on
//  the shader side; see `docs/PRECISION.md` for the (larger) story on the
//  backend side, because WGSL as of this writing has no `f64`.
//
//  The *rendering* shaders deliberately stay in `f32`: display precision is
//  bounded by the framebuffer, not by the physics.
//
//  LATTICE
//  -------
//  A cubic lattice of nx x ny x nz cells with spacing `dx` and periodic
//  boundary conditions in all three directions.  Linear index is
//      i = (z * ny + y) * nx + x
//  so that consecutive threads in a workgroup (which vary in x) touch
//  consecutive addresses.

struct SimParams {
    // --- lattice -----------------------------------------------------------
    nx: u32,
    ny: u32,
    nz: u32,
    _pad_u0: u32,

    dx: flt,
    inv_dx: flt,
    dt: flt,
    eta: flt,                // phenomenological wall friction  [1/length]

    // --- scalar potential (see potential.wgsl) -----------------------------
    lambda: flt,             // quartic coupling
    eps: flt,                // vacuum energy difference V(0) - V(phi_b) > 0
    phi_b: flt,              // broken-phase (true vacuum) field value
    m2: flt,                 // derived: V''(0)

    delta: flt,              // derived: cubic coefficient
    a_rad: flt,              // radiation constant, e_fluid = a_rad * T^4
    wall_width: flt,         // derived: tanh wall thickness l_w
    p_floor: flt,            // pressure floor for robustness

    // --- primitive-recovery safety valves -----------------------------------
    e_floor: flt,            // floor on lab-frame fluid energy density
    z_max: flt,              // |Z| <= z_max * E  (keeps |v| < 1)
    e_ref: flt,              // reference fluid energy density (initial state)
    t_ref: flt,              // reference temperature (initial state)

    time: flt,               // simulation time
    vis_gain: flt,           // scale for the visualisation texture
    _pad0: flt,
    _pad1: flt,
}

// Runge-Kutta stage coefficients.  Every stage computes
//     U_out = a0 * U_n  +  a1 * (U_in + dt_stage * L(U_in))
// which covers all three stages of SSP-RK3 (see docs/NUMERICS.md).
struct StageParams {
    a0: flt,
    a1: flt,
    dt_stage: flt,
    _pad: flt,
}

@group(0) @binding(0) var<uniform> P: SimParams;

// ---------------------------------------------------------------------------
//  Indexing
// ---------------------------------------------------------------------------

fn grid_dims() -> vec3<i32> {
    return vec3<i32>(i32(P.nx), i32(P.ny), i32(P.nz));
}

fn wrap1(i: i32, n: i32) -> i32 {
    // Periodic wrap.  The stencils reach at most 2 cells, so a single
    // conditional pair is enough and is cheaper than a modulo.
    var r = i;
    if (r < 0) { r += n; }
    if (r < 0) { r += n; }
    if (r >= n) { r -= n; }
    if (r >= n) { r -= n; }
    return r;
}

fn cell_index(c: vec3<i32>) -> u32 {
    let n = grid_dims();
    let w = vec3<i32>(wrap1(c.x, n.x), wrap1(c.y, n.y), wrap1(c.z, n.z));
    return u32((w.z * n.y + w.y) * n.x + w.x);
}

fn cell_count() -> u32 {
    return P.nx * P.ny * P.nz;
}

// Unit offset along axis 0/1/2.
fn axis_offset(axis: i32) -> vec3<i32> {
    return vec3<i32>(
        select(0, 1, axis == 0),
        select(0, 1, axis == 1),
        select(0, 1, axis == 2),
    );
}

// ---------------------------------------------------------------------------
//  Slope limiting (used by the MUSCL reconstruction in step.wgsl)
// ---------------------------------------------------------------------------

fn minmod3(a: flt, b: flt, c: flt) -> flt {
    let sa = sign(a);
    if (sa != sign(b) || sa != sign(c)) {
        return flt(0.0);
    }
    return sa * min(abs(a), min(abs(b), abs(c)));
}

// Monotonised-central limiter: second-order accurate in smooth regions,
// total-variation-diminishing across shocks.  Swap `minmod3(...)` for
// `minmod3(dm, dm, dp)` to get the more diffusive plain minmod limiter.
fn limit_slope(dm: flt, dp: flt) -> flt {
    return minmod3(flt(0.5) * (dm + dp), flt(2.0) * dm, flt(2.0) * dp);
}

fn limit_slope4(dm: vec4<flt>, dp: vec4<flt>) -> vec4<flt> {
    return vec4<flt>(
        limit_slope(dm.x, dp.x),
        limit_slope(dm.y, dp.y),
        limit_slope(dm.z, dp.z),
        limit_slope(dm.w, dp.w),
    );
}
