// ============================================================================
//  step.wgsl -- one SSP-RK3 stage of the coupled scalar-field + fluid system.
// ============================================================================
//
//  MODEL (units c = hbar = k_B = 1, metric signature +,-,-,-)
//  ---------------------------------------------------------
//  A real scalar order parameter phi coupled to a relativistic perfect fluid
//  through a phenomenological friction term of strength eta:
//
//      d_mu d^mu phi + dV/dphi = - eta u^mu d_mu phi                      (1)
//
//  Because T^{mu nu}_phi obeys  d_mu T^{mu nu}_phi = (box phi + V') d^nu phi,
//  equation (1) makes the field lose exactly
//
//      d_mu T^{mu nu}_phi = - eta (u^lambda d_lambda phi) d^nu phi
//
//  which the fluid must pick up, since the *total* stress tensor is conserved:
//
//      d_mu T^{mu nu}_fluid = + eta (u^lambda d_lambda phi) d^nu phi      (2)
//
//  Writing (1) as a first-order system in pi = dphi/dt, and (2) in terms of the
//  lab-frame densities E = T^{00} and Z^i = T^{0i} (see eos.wgsl), the six
//  evolved fields obey
//
//      dphi/dt = pi
//      dpi /dt = laplacian(phi) - dV/dphi - eta D
//      dE  /dt = -div(Z)                  + eta D pi
//      dZ^i/dt = -d_j (Z^i v^j + p d^ij)  - eta D d_i phi
//
//  with  D = u^mu d_mu phi = W (pi + v . grad phi).  The friction terms cancel
//  pairwise between the field and fluid sectors, so total energy is conserved
//  up to discretisation error -- the live "energy drift" readout is a direct
//  check of that.
//
//  DISCRETISATION
//  --------------
//  * Field sector: second-order central differences (7-point Laplacian).
//  * Fluid sector: second-order MUSCL reconstruction of the primitive
//    variables with a monotonised-central limiter, and an HLL approximate
//    Riemann solver at each cell face.  Shock capturing matters here: the
//    interesting output of the simulation is the shock in front of a
//    deflagration and the sound waves left behind after the bubbles merge.
//  * Time: strong-stability-preserving RK3 (see docs/NUMERICS.md).
//
//  Each stage evaluates  U_out = a0 U^n + a1 (U_in + dt L(U_in)).

//!include common
//!include potential
//!include eos

@group(0) @binding(1) var<uniform> S: StageParams;

@group(1) @binding(0) var<storage, read>       field_n   : array<vec2<flt>>;  // U^n
@group(1) @binding(1) var<storage, read>       fluid_n   : array<vec4<flt>>;
@group(1) @binding(2) var<storage, read>       field_in  : array<vec2<flt>>;  // stage input
@group(1) @binding(3) var<storage, read>       fluid_in  : array<vec4<flt>>;
@group(1) @binding(4) var<storage, read>       prim      : array<vec4<flt>>;
@group(1) @binding(5) var<storage, read_write> field_out : array<vec2<flt>>;
@group(1) @binding(6) var<storage, read_write> fluid_out : array<vec4<flt>>;

// Keep a reconstructed face state inside the physical domain: positive
// pressure and sub-luminal velocity.
fn sanitise(q: vec4<flt>) -> vec4<flt> {
    let p = max(q.w, P.p_floor);
    var v = q.xyz;
    let v2 = dot(v, v);
    let vmax2 = P.z_max * P.z_max;
    if (v2 > vmax2) {
        v = v * sqrt(vmax2 / max(v2, flt(1e-30)));
    }
    return vec4<flt>(v, p);
}

// HLL approximate Riemann solver.  Using min(.,0) / max(.,0) on the signal
// speeds makes the formula degrade gracefully to pure upwinding for
// supersonic faces.
fn hll_flux(q_l: vec4<flt>, phi_l: flt, q_r: vec4<flt>, phi_r: flt, axis: i32) -> vec4<flt> {
    let u_l = eos_cons_from_prim(q_l, phi_l);
    let u_r = eos_cons_from_prim(q_r, phi_r);
    let f_l = eos_flux(q_l, u_l, axis);
    let f_r = eos_flux(q_r, u_r, axis);

    let lam_l = eos_wave_speeds(q_l, phi_l, axis);
    let lam_r = eos_wave_speeds(q_r, phi_r, axis);
    let s_min = min(min(lam_l.x, lam_r.x), flt(0.0));
    let s_max = max(max(lam_l.y, lam_r.y), flt(0.0));

    let den = max(s_max - s_min, flt(1e-12));
    return (s_max * f_l - s_min * f_r + s_min * s_max * (u_r - u_l)) / den;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.nx || gid.y >= P.ny || gid.z >= P.nz) {
        return;
    }
    let c = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));
    let i0 = cell_index(c);

    let f0 = field_in[i0];
    let phi0 = f0.x;
    let pi0 = f0.y;
    let q0 = sanitise(prim[i0]);

    var lap_sum = flt(0.0);            // sum of (phi_- + phi_+ - 2 phi_0)
    var grad = vec3<flt>(flt(0.0), flt(0.0), flt(0.0));
    var flux_div = vec4<flt>(flt(0.0), flt(0.0), flt(0.0), flt(0.0));

    for (var axis = 0; axis < 3; axis = axis + 1) {
        let e1 = axis_offset(axis);
        let e2 = e1 + e1;

        let im1 = cell_index(c - e1);
        let ip1 = cell_index(c + e1);
        let im2 = cell_index(c - e2);
        let ip2 = cell_index(c + e2);

        // ---- scalar field: 7-point Laplacian and central gradient ----------
        let phi_m = field_in[im1].x;
        let phi_p = field_in[ip1].x;
        lap_sum = lap_sum + (phi_m + phi_p - flt(2.0) * phi0);
        grad[axis] = flt(0.5) * (phi_p - phi_m) * P.inv_dx;

        // ---- fluid: MUSCL reconstruction + HLL fluxes ----------------------
        let qm2 = sanitise(prim[im2]);
        let qm1 = sanitise(prim[im1]);
        let qp1 = sanitise(prim[ip1]);
        let qp2 = sanitise(prim[ip2]);

        let slope_m1 = limit_slope4(qm1 - qm2, q0 - qm1);
        let slope_0  = limit_slope4(q0 - qm1, qp1 - q0);
        let slope_p1 = limit_slope4(qp1 - q0, qp2 - qp1);

        // Face i-1/2
        let f_minus = hll_flux(
            sanitise(qm1 + flt(0.5) * slope_m1), phi_m,
            sanitise(q0 - flt(0.5) * slope_0), phi0,
            axis,
        );
        // Face i+1/2
        let f_plus = hll_flux(
            sanitise(q0 + flt(0.5) * slope_0), phi0,
            sanitise(qp1 - flt(0.5) * slope_p1), phi_p,
            axis,
        );

        flux_div = flux_div + (f_plus - f_minus) * P.inv_dx;
    }

    let laplacian = lap_sum * P.inv_dx * P.inv_dx;

    // ---- friction coupling -------------------------------------------------
    let v0 = q0.xyz;
    let gam = flt(1.0) / sqrt(max(flt(1.0) - dot(v0, v0), flt(1e-12)));
    let dphi_proper = gam * (pi0 + dot(v0, grad));      // D = u^mu d_mu phi
    let drag = P.eta * dphi_proper;

    // ---- right-hand sides --------------------------------------------------
    let d_phi = pi0;
    let d_pi = laplacian - dpotential(phi0) - drag;

    var d_fluid = -flux_div;
    d_fluid.x = d_fluid.x + drag * pi0;
    d_fluid.y = d_fluid.y - drag * grad.x;
    d_fluid.z = d_fluid.z - drag * grad.y;
    d_fluid.w = d_fluid.w - drag * grad.z;

    // ---- SSP-RK3 stage combination ----------------------------------------
    let field_upd = f0 + S.dt_stage * vec2<flt>(d_phi, d_pi);
    let fluid_upd = fluid_in[i0] + S.dt_stage * d_fluid;

    field_out[i0] = S.a0 * field_n[i0] + S.a1 * field_upd;
    fluid_out[i0] = S.a0 * fluid_n[i0] + S.a1 * fluid_upd;
}
