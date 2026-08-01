// ============================================================================
//  nucleate.wgsl -- stamp super-critical bubbles of broken phase into the field.
// ============================================================================
//
//  A bubble is written as the planar-wall (kink) profile of the degenerate
//  double well, wrapped onto a sphere of radius r0:
//
//      phi(r) = (phi_b / 2) * [ 1 - tanh( (r - r0) / l_w ) ]
//
//  with wall thickness  l_w = 2 sqrt(2 / lambda), which is the exact kink width
//  of V = (lambda/4) phi^2 (phi - phi_b)^2.  For eps > 0 the true wall is
//  slightly thinner, but the profile relaxes to the correct one within a few
//  light-crossing times of the wall, long before the bubble has grown.
//
//  Seeding a bubble at r0 > R_crit = 2 sigma / eps guarantees it grows rather
//  than collapses; the host reports R_crit in the UI.
//
//  Only the field is modified.  The latent heat that ends up in the plasma is
//  produced self-consistently by the friction term as the wall moves, not
//  injected here.  Note that stamping a bubble does add its surface and vacuum
//  energy to the box by hand, so the total-energy readout steps at each
//  nucleation event; the host re-baselines the drift measurement afterwards.
//
//  Overlapping bubbles are merged with max(), which is the correct behaviour
//  for a scalar order parameter: the broken region is the union.

//!include common

struct Bubble {
    // Centre in lattice coordinates (cells), and seed radius in cells.
    cx: flt,
    cy: flt,
    cz: flt,
    r0: flt,
}

// The bubble count lives in its own uniform rather than in SimParams so that
// the nucleation dispatch can be recorded in the same submission as the
// following time steps without the two parameter writes racing.
struct NucleationBatch {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(1) @binding(0) var<storage, read_write> field_io : array<vec2<flt>>;
@group(1) @binding(1) var<storage, read>       bubbles  : array<Bubble>;
@group(1) @binding(2) var<uniform>             batch    : NucleationBatch;

// Minimum-image separation on a periodic lattice.
fn min_image(d: flt, n: flt) -> flt {
    var r = d;
    if (r > flt(0.5) * n) { r = r - n; }
    if (r < flt(-0.5) * n) { r = r + n; }
    return r;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.nx || gid.y >= P.ny || gid.z >= P.nz) {
        return;
    }
    let i = (gid.z * P.ny + gid.y) * P.nx + gid.x;
    let pos = vec3<flt>(flt(gid.x), flt(gid.y), flt(gid.z));
    let n = vec3<flt>(flt(P.nx), flt(P.ny), flt(P.nz));

    var phi = field_io[i].x;
    let pi_old = field_io[i].y;

    for (var b = 0u; b < batch.count; b = b + 1u) {
        let bub = bubbles[b];
        let d = vec3<flt>(
            min_image(pos.x - bub.cx, n.x),
            min_image(pos.y - bub.cy, n.y),
            min_image(pos.z - bub.cz, n.z),
        );
        let r = length(d) * P.dx;
        let profile = flt(0.5) * P.phi_b
            * (flt(1.0) - tanh((r - bub.r0 * P.dx) / P.wall_width));
        phi = max(phi, profile);
    }

    field_io[i] = vec2<flt>(phi, pi_old);
}
