// ============================================================================
//  vis.wgsl -- pack the live simulation state into a 3D texture for rendering.
// ============================================================================
//
//  This is the *only* bridge between the solver and the renderer, and it stays
//  entirely on the device: a compute pass writes an rgba16float 3D storage
//  texture which the ray-marching fragment shader samples (with hardware
//  trilinear filtering) in the very same frame.  Simulation state is never
//  copied to host memory.
//
//  Channels
//      r : phi / phi_b        order parameter; 0 = symmetric, 1 = broken
//      g : |v|                fluid three-velocity magnitude
//      b : T / T_ref - 1      temperature contrast (signed, small)
//      a : (E + p) |v|^2 / e_ref
//                             fluid kinetic energy density -- the quantity that
//                             sources gravitational waves, and the one in which
//                             the sound-wave field is most visible
//
//  Storing *contrasts* rather than absolute values matters: rgba16float keeps
//  ~11 bits of mantissa, which is plenty for a fluctuation of relative size
//  1e-3 but would be marginal for T itself.

//!include common
//!include eos

@group(1) @binding(0) var<storage, read> field_in : array<vec2<flt>>;
@group(1) @binding(1) var<storage, read> fluid_in : array<vec4<flt>>;
@group(1) @binding(2) var<storage, read> prim     : array<vec4<flt>>;
@group(1) @binding(3) var vis_tex : texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.nx || gid.y >= P.ny || gid.z >= P.nz) {
        return;
    }
    let i = (gid.z * P.ny + gid.y) * P.nx + gid.x;

    let phi = field_in[i].x;
    let q = prim[i];
    let v = q.xyz;
    let p = q.w;
    let e_lab = max(fluid_in[i].x, P.e_floor);

    let speed = length(v);
    let temp = eos_temperature(p, phi);
    let kinetic = (e_lab + p) * dot(v, v);

    textureStore(vis_tex, vec3<i32>(gid), vec4<f32>(
        f32(phi / P.phi_b),
        f32(speed),
        f32(temp / P.t_ref - flt(1.0)),
        f32(kinetic / P.e_ref),
    ));
}
