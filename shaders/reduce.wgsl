// ============================================================================
//  reduce.wgsl -- global diagnostics, computed on the GPU.
// ============================================================================
//
//  Two dispatches:
//      pass1: 1024 workgroups x 256 threads, grid-striding over the lattice,
//             each workgroup emitting one partial sum (plus one atomic max).
//      pass2: a single workgroup folding the 1024 partials into the final value.
//
//  The result is 32 bytes.  Those 32 bytes are the *only* thing that is ever
//  copied back to the host, and even that happens through a non-blocking
//  mapped staging buffer read one frame late, purely to print numbers in the
//  UI.  The lattice itself never leaves device memory.
//
//  Accumulated quantities (all summed over cells, then divided by cell count
//  on the host):
//      x : total energy density  1/2 pi^2 + 1/2 |grad phi|^2 + V(phi) + E
//      y : broken-phase volume fraction
//      z : fluid kinetic energy density  (E + p) v^2  = w W^2 v^2
//      w : fluid lab-frame energy density E
//  plus max |v| over the box, carried as the bit pattern of a non-negative f32
//  (IEEE-754 order-preserving for non-negative values, so atomicMax on u32
//  gives the right answer).

//!include common
//!include potential
//!include eos

const N_PARTIALS: u32 = 1024u;
const WG_SIZE: u32 = 256u;

struct Diagnostics {
    sums: vec4<f32>,
    max_v_bits: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(1) @binding(0) var<storage, read>       field_in : array<vec2<flt>>;
@group(1) @binding(1) var<storage, read>       fluid_in : array<vec4<flt>>;
@group(1) @binding(2) var<storage, read>       prim     : array<vec4<flt>>;
@group(1) @binding(3) var<storage, read_write> partials : array<vec4<flt>>;
@group(1) @binding(4) var<storage, read_write> diag     : Diagnostics;

var<workgroup> shared_sum: array<vec4<flt>, WG_SIZE>;
var<workgroup> shared_max: array<flt, WG_SIZE>;

fn cell_coords(i: u32) -> vec3<i32> {
    let x = i % P.nx;
    let y = (i / P.nx) % P.ny;
    let z = i / (P.nx * P.ny);
    return vec3<i32>(i32(x), i32(y), i32(z));
}

fn cell_contribution(i: u32) -> vec4<flt> {
    let c = cell_coords(i);
    let f = field_in[i];
    let phi = f.x;
    let pi = f.y;

    var grad2 = flt(0.0);
    for (var axis = 0; axis < 3; axis = axis + 1) {
        let e1 = axis_offset(axis);
        let d = flt(0.5) * (field_in[cell_index(c + e1)].x
                          - field_in[cell_index(c - e1)].x) * P.inv_dx;
        grad2 = grad2 + d * d;
    }

    let e_lab = max(fluid_in[i].x, P.e_floor);
    let q = prim[i];
    let v2 = dot(q.xyz, q.xyz);

    let e_total = flt(0.5) * pi * pi + flt(0.5) * grad2 + potential(phi) + e_lab;
    let broken = select(flt(0.0), flt(1.0), phi > flt(0.5) * P.phi_b);
    let kinetic = (e_lab + q.w) * v2;

    return vec4<flt>(e_total, broken, kinetic, e_lab);
}

@compute @workgroup_size(256, 1, 1)
fn reduce_pass1(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let total = cell_count();
    let stride = WG_SIZE * N_PARTIALS;

    var acc = vec4<flt>(flt(0.0), flt(0.0), flt(0.0), flt(0.0));
    var vmax = flt(0.0);

    var i = gid.x;
    while (i < total) {
        acc = acc + cell_contribution(i);
        vmax = max(vmax, length(prim[i].xyz));
        i = i + stride;
    }

    shared_sum[lid] = acc;
    shared_max[lid] = vmax;
    workgroupBarrier();

    for (var s = WG_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            shared_sum[lid] = shared_sum[lid] + shared_sum[lid + s];
            shared_max[lid] = max(shared_max[lid], shared_max[lid + s]);
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        partials[wid.x] = shared_sum[0];
        atomicMax(&diag.max_v_bits, bitcast<u32>(f32(shared_max[0])));
    }
}

@compute @workgroup_size(256, 1, 1)
fn reduce_pass2(@builtin(local_invocation_index) lid: u32) {
    var acc = vec4<flt>(flt(0.0), flt(0.0), flt(0.0), flt(0.0));
    var i = lid;
    while (i < N_PARTIALS) {
        acc = acc + partials[i];
        i = i + WG_SIZE;
    }
    shared_sum[lid] = acc;
    workgroupBarrier();

    for (var s = WG_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            shared_sum[lid] = shared_sum[lid] + shared_sum[lid + s];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        let t = shared_sum[0];
        diag.sums = vec4<f32>(f32(t.x), f32(t.y), f32(t.z), f32(t.w));
    }
}
