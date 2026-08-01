// ============================================================================
//  primitives.wgsl -- conserved (E, Z) -> primitive (v, p), once per RK stage.
// ============================================================================
//
//  Done as its own pass because the MUSCL reconstruction in step.wgsl needs the
//  primitives of 13 neighbouring cells; recovering them there would repeat the
//  (branchy, square-root heavy) inversion ~13x per cell.

//!include common
//!include eos

@group(1) @binding(0) var<storage, read>       field_in : array<vec2<flt>>;
@group(1) @binding(1) var<storage, read>       fluid_in : array<vec4<flt>>;
@group(1) @binding(2) var<storage, read_write> prim_out : array<vec4<flt>>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.nx || gid.y >= P.ny || gid.z >= P.nz) {
        return;
    }
    let i = (gid.z * P.ny + gid.y) * P.nx + gid.x;
    prim_out[i] = eos_prim_from_cons(fluid_in[i], field_in[i].x);
}
