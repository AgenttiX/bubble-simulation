// ============================================================================
//  init.wgsl -- initial condition: homogeneous symmetric phase at rest.
// ============================================================================
//
//  phi = 0 (false vacuum), pi = 0, fluid at rest with energy density e_ref.
//  At rest, E = T^{00} = e_ref and Z = 0.
//
//  Bubbles are *not* nucleated here.  Thermal/quantum nucleation is a rare
//  event that a lattice simulation of this size cannot resolve, so -- as in
//  published simulations of this system -- bubbles are seeded explicitly by
//  nucleate.wgsl according to a schedule chosen on the host.

//!include common

@group(1) @binding(0) var<storage, read_write> field_out : array<vec2<flt>>;
@group(1) @binding(1) var<storage, read_write> fluid_out : array<vec4<flt>>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.nx || gid.y >= P.ny || gid.z >= P.nz) {
        return;
    }
    let i = (gid.z * P.ny + gid.y) * P.nx + gid.x;
    field_out[i] = vec2<flt>(flt(0.0), flt(0.0));
    fluid_out[i] = vec4<flt>(P.e_ref, flt(0.0), flt(0.0), flt(0.0));
}
