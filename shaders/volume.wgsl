// ============================================================================
//  volume.wgsl -- interactive volume renderer for the live lattice.
// ============================================================================
//
//  A single full-screen triangle; every pixel marches a ray through the 3D
//  texture that vis.wgsl wrote earlier in the same frame.  Nothing is read back
//  to the host, and no geometry is generated on the CPU.
//
//  Two things are drawn simultaneously:
//
//    1. The bubble walls, as a shaded isosurface of the order parameter at
//       phi = iso * phi_b.  Crossings are detected between consecutive samples
//       and refined by linear interpolation; the normal comes from the
//       gradient of the trilinearly-filtered texture.
//
//    2. The plasma, as emissive volume: sound waves, the compression shell in
//       front of a deflagration, and the reheated interior all show up here.
//
//  Rendering deliberately stays in f32 regardless of the solver's precision --
//  the framebuffer is 8 bits per channel, so there is nothing to gain.

struct View {
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,      // xyz = eye position in world space
    box_half: vec4<f32>,     // xyz = half extents of the lattice box, w = cell size
    grid: vec4<f32>,         // xyz = nx, ny, nz
    render0: vec4<f32>,      // iso_level, wall_opacity, samples_per_cell, exposure
    render1: vec4<f32>,      // field_mode, clip_axis, clip_pos, colormap
    render2: vec4<f32>,      // field_gain, absorption, show_iso, show_volume
    render3: vec4<f32>,      // frame_jitter, show_box, background, unused
}

@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var vis_tex: texture_3d<f32>;
@group(0) @binding(2) var vis_samp: sampler;

const MAX_STEPS: i32 = 2048;

// ---------------------------------------------------------------------------
//  Colour maps (11 control points each, linearly interpolated)
// ---------------------------------------------------------------------------

var<private> VIRIDIS: array<vec3<f32>, 11> = array<vec3<f32>, 11>(
    vec3<f32>(0.267, 0.005, 0.329), vec3<f32>(0.283, 0.141, 0.458),
    vec3<f32>(0.254, 0.265, 0.530), vec3<f32>(0.207, 0.372, 0.553),
    vec3<f32>(0.164, 0.471, 0.558), vec3<f32>(0.128, 0.567, 0.551),
    vec3<f32>(0.135, 0.659, 0.518), vec3<f32>(0.267, 0.749, 0.441),
    vec3<f32>(0.478, 0.821, 0.318), vec3<f32>(0.741, 0.873, 0.150),
    vec3<f32>(0.993, 0.906, 0.144),
);

var<private> INFERNO: array<vec3<f32>, 11> = array<vec3<f32>, 11>(
    vec3<f32>(0.001, 0.000, 0.014), vec3<f32>(0.078, 0.044, 0.214),
    vec3<f32>(0.225, 0.036, 0.388), vec3<f32>(0.373, 0.074, 0.432),
    vec3<f32>(0.522, 0.128, 0.420), vec3<f32>(0.665, 0.182, 0.370),
    vec3<f32>(0.798, 0.280, 0.270), vec3<f32>(0.902, 0.413, 0.144),
    vec3<f32>(0.969, 0.579, 0.032), vec3<f32>(0.988, 0.762, 0.135),
    vec3<f32>(0.988, 0.998, 0.645),
);

// Signed map for quantities that can go either way, with zero at t = 0.5.
//
// Unlike a printed diverging map, the centre is near *black* rather than white.
// In an emissive volume the centre colour is what the vast, weakly perturbed
// bulk of the box paints, and a white centre turns that bulk into an opaque
// milky fog that hides the structure entirely.  Fading to black makes zero
// contrast literally invisible, so only real departures are drawn: cold and
// compressed reads blue, hot and rarefied reads orange.
var<private> SIGNED: array<vec3<f32>, 11> = array<vec3<f32>, 11>(
    vec3<f32>(0.30, 0.55, 1.00), vec3<f32>(0.22, 0.42, 0.85),
    vec3<f32>(0.15, 0.30, 0.66), vec3<f32>(0.09, 0.19, 0.45),
    vec3<f32>(0.04, 0.09, 0.24), vec3<f32>(0.02, 0.02, 0.03),
    vec3<f32>(0.26, 0.09, 0.04), vec3<f32>(0.48, 0.17, 0.06),
    vec3<f32>(0.72, 0.29, 0.08), vec3<f32>(0.92, 0.48, 0.14),
    vec3<f32>(1.00, 0.72, 0.32),
);

fn ramp(t: f32, which: i32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0) * 10.0;
    let i0 = i32(floor(x));
    let i1 = min(i0 + 1, 10);
    let f = x - floor(x);
    if (which == 0) {
        return mix(VIRIDIS[i0], VIRIDIS[i1], f);
    } else if (which == 1) {
        return mix(INFERNO[i0], INFERNO[i1], f);
    }
    return mix(SIGNED[i0], SIGNED[i1], f);
}

// ---------------------------------------------------------------------------
//  Geometry helpers
// ---------------------------------------------------------------------------

// Sign-preserving reciprocal; rays exactly parallel to an axis get a huge but
// finite slope, which the slab test below handles correctly.
fn safe_inv(x: f32) -> f32 {
    let s = select(1.0, -1.0, x < 0.0);
    return s / max(abs(x), 1e-9);
}

// Slab test against the axis-aligned box centred on the origin.
// Returns (t_near, t_far); a miss has t_near > t_far.
fn intersect_box(ro: vec3<f32>, inv_rd: vec3<f32>, half: vec3<f32>) -> vec2<f32> {
    let t0 = (-half - ro) * inv_rd;
    let t1 = (half - ro) * inv_rd;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(t_near, t_far);
}

fn to_texcoord(p: vec3<f32>) -> vec3<f32> {
    return (p / V.box_half.xyz) * 0.5 + vec3<f32>(0.5);
}

fn sample_state(p: vec3<f32>) -> vec4<f32> {
    return textureSampleLevel(vis_tex, vis_samp, to_texcoord(p), 0.0);
}

// Gradient of the order parameter, in world units.
fn grad_phi(p: vec3<f32>) -> vec3<f32> {
    let h = V.box_half.w;   // one cell
    let dx = vec3<f32>(h, 0.0, 0.0);
    let dy = vec3<f32>(0.0, h, 0.0);
    let dz = vec3<f32>(0.0, 0.0, h);
    return vec3<f32>(
        sample_state(p + dx).r - sample_state(p - dx).r,
        sample_state(p + dy).r - sample_state(p - dy).r,
        sample_state(p + dz).r - sample_state(p - dz).r,
    ) / (2.0 * h);
}

fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + vec3<f32>(dot(p3, p3.yzx + vec3<f32>(33.33)));
    return fract((p3.x + p3.y) * p3.z);
}

// ---------------------------------------------------------------------------
//  Field selection
// ---------------------------------------------------------------------------

struct Mapped {
    t: f32,          // position on the colour ramp, 0..1
    intensity: f32,  // emission/opacity weight, >= 0
    ramp_id: i32,
}

fn map_field(s: vec4<f32>) -> Mapped {
    let mode = i32(V.render1.x);
    let gain = V.render2.x;
    var m: Mapped;
    m.ramp_id = i32(V.render1.w);

    if (mode == 0) {
        // Order parameter: 0 = symmetric, 1 = broken.
        let x = clamp(s.r, 0.0, 1.0);
        m.t = x;
        m.intensity = x * gain;
    } else if (mode == 1) {
        // Fluid speed.
        let x = clamp(s.g * gain, 0.0, 1.0);
        m.t = x;
        m.intensity = x;
    } else if (mode == 2) {
        // Temperature contrast: signed, so use the diverging ramp.
        let x = clamp(s.b * gain, -1.0, 1.0);
        m.t = 0.5 + 0.5 * x;
        m.intensity = abs(x);
        m.ramp_id = 2;
    } else {
        // Fluid kinetic energy density.
        let x = clamp(s.a * gain, 0.0, 1.0);
        m.t = x;
        m.intensity = x;
    }
    return m;
}

// ---------------------------------------------------------------------------
//  Vertex stage: one oversized triangle covering the screen
// ---------------------------------------------------------------------------

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    var out: VsOut;
    out.ndc = vec2<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0);
    out.pos = vec4<f32>(out.ndc, 0.0, 1.0);
    return out;
}

// ---------------------------------------------------------------------------
//  Fragment stage: the ray march
// ---------------------------------------------------------------------------

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let ro = V.cam_pos.xyz;

    // Unproject the far plane to get the ray direction (reverse-Z friendly:
    // wgpu clip space has z in [0,1], so z = 1 is the far plane).
    let far = V.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let rd = normalize(far.xyz / far.w - ro);

    // Background: a soft vertical gradient so the box reads as a 3D object.
    let horizon = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
    var color = mix(vec3<f32>(0.015, 0.017, 0.028), vec3<f32>(0.045, 0.055, 0.085), horizon)
              * V.render3.z;

    let inv_rd = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));
    let hit = intersect_box(ro, inv_rd, V.box_half.xyz);
    var t_near = max(hit.x, 0.0);
    var t_far = hit.y;
    if (t_near >= t_far) {
        return vec4<f32>(color, 1.0);
    }

    // Cutaway: clip the ray against a movable axis-aligned half-space so the
    // interior of the box can be inspected.
    let clip_axis = i32(V.render1.y);
    if (clip_axis >= 0) {
        let n_axis = vec3<f32>(
            select(0.0, 1.0, clip_axis == 0),
            select(0.0, 1.0, clip_axis == 1),
            select(0.0, 1.0, clip_axis == 2),
        );
        let plane = V.render1.z * V.box_half[clip_axis];
        let denom = dot(rd, n_axis);
        let dist = plane - dot(ro, n_axis);
        // Keep the half-space  dot(p, n) <= plane.
        if (abs(denom) < 1e-6) {
            if (dist < 0.0) { return vec4<f32>(color, 1.0); }
        } else {
            let t_plane = dist / denom;
            if (denom > 0.0) {
                t_far = min(t_far, t_plane);
            } else {
                t_near = max(t_near, t_plane);
            }
            if (t_near >= t_far) { return vec4<f32>(color, 1.0); }
        }
    }

    // --- box wireframe ------------------------------------------------------
    if (V.render3.y > 0.5) {
        let w = 0.004 * V.box_half.w * V.grid.x;
        var edge = 0.0;
        for (var k = 0; k < 2; k = k + 1) {
            let p = ro + rd * select(t_far, t_near, k == 0);
            let d = V.box_half.xyz - abs(p);
            // A point on a face has one small component of `d`; a point on an
            // edge has two.  So the *median* of the three is the edge distance.
            let median = max(min(d.x, d.y), min(max(d.x, d.y), d.z));
            edge = max(edge, 1.0 - smoothstep(0.0, w, median));
        }
        color = color + vec3<f32>(0.20, 0.26, 0.34) * edge;
    }

    // --- march --------------------------------------------------------------
    let cell = V.box_half.w;
    let samples_per_cell = max(V.render0.z, 0.25);
    var ds = cell / samples_per_cell;
    let span = t_far - t_near;
    if (span / ds > f32(MAX_STEPS)) {
        ds = span / f32(MAX_STEPS);
    }
    let n_steps = min(i32(ceil(span / ds)), MAX_STEPS);

    // Jitter the first sample to trade banding for noise.
    let jitter = hash12(in.pos.xy + vec2<f32>(V.render3.x));

    let iso = V.render0.x;
    let wall_opacity = V.render0.y;
    let exposure = V.render0.w;
    let absorption = V.render2.y;
    let show_iso = V.render2.z > 0.5;
    let show_vol = V.render2.w > 0.5;

    var accum = vec3<f32>(0.0);
    var alpha = 0.0;

    var t = t_near + jitter * ds;
    var prev_phi = sample_state(ro + rd * t_near).r;
    var prev_t = t_near;

    for (var i = 0; i < n_steps; i = i + 1) {
        if (t > t_far || alpha > 0.995) { break; }
        let p = ro + rd * t;
        let s = sample_state(p);

        // --- bubble wall isosurface ---------------------------------------
        //
        // The test is `<= 0`, not `< 0`, and that matters.  Hardware trilinear
        // filtering computes its weights in fixed point, so a sample lands
        // *exactly* on the isolevel far more often than continuous arithmetic
        // would suggest -- around one pixel in a hundred.  With a strict
        // inequality those samples register no crossing at all and punch
        // visible pinholes through the wall.  Requiring `d_prev != 0` attributes
        // each crossing to exactly one interval, so nothing is counted twice.
        let d_prev = prev_phi - iso;
        let d_cur = s.r - iso;
        if (show_iso && d_prev != 0.0 && d_prev * d_cur <= 0.0) {
            let denom = s.r - prev_phi;
            let frac = select(0.5, (iso - prev_phi) / denom, abs(denom) > 1e-6);
            let t_hit = mix(prev_t, t, clamp(frac, 0.0, 1.0));
            let p_hit = ro + rd * t_hit;

            // The broken phase is where phi is large, so the outward normal of
            // the bubble points down the gradient.
            let g = grad_phi(p_hit);
            let n = normalize(-g - rd * 1e-6);

            let hit_state = sample_state(p_hit);
            // Tint the wall by how hard it is driving the plasma.  Scaled on its
            // own, not by the volume field's gain: 0.25 c saturates the tint,
            // which spans the range walls actually reach.
            let drive = clamp(hit_state.g * 4.0, 0.0, 1.0);
            let base = mix(vec3<f32>(0.16, 0.62, 0.72), vec3<f32>(1.00, 0.72, 0.28), drive);

            let l1 = normalize(vec3<f32>(0.55, 0.72, 0.42));
            let l2 = normalize(vec3<f32>(-0.5, -0.2, -0.7));
            let view = -rd;
            let diff = max(dot(n, l1), 0.0) + 0.35 * max(dot(n, l2), 0.0);
            let h = normalize(l1 + view + vec3<f32>(0.0, 1e-5, 0.0));
            let spec = pow(max(dot(n, h), 0.0), 48.0);
            let fres = pow(1.0 - abs(dot(n, view)), 3.0);

            let shaded = base * (0.22 + 0.85 * diff)
                       + vec3<f32>(1.0) * spec * 0.35
                       + base * fres * 0.9;

            let a = wall_opacity;
            accum = accum + (1.0 - alpha) * shaded * a;
            alpha = alpha + (1.0 - alpha) * a;
        }

        // --- emissive plasma ------------------------------------------------
        if (show_vol) {
            let m = map_field(s);
            if (m.intensity > 1e-4) {
                // Opacity follows intensity *squared* while colour stays
                // linear.  Without that, a weak but box-filling background --
                // the ambient reheating left behind by sound waves, say --
                // accumulates into a milky veil that hides everything, because
                // a small opacity times a long path is still a large optical
                // depth.  Squaring suppresses the background hard while leaving
                // saturated features exactly as opaque as before.
                //
                // `ds` is in world units, where the box spans 1.0, so
                // `absorption` is the optical depth across the whole box at
                // unit intensity, independent of the lattice size.
                let weight = m.intensity * m.intensity;
                let a = 1.0 - exp(-absorption * weight * ds);
                let emit = ramp(m.t, m.ramp_id) * exposure;
                accum = accum + (1.0 - alpha) * emit * a;
                alpha = alpha + (1.0 - alpha) * a;
            }
        }

        prev_phi = s.r;
        prev_t = t;
        t = t + ds;
    }

    color = color * (1.0 - alpha) + accum;

    // Filmic-ish tonemap keeps the bright shock fronts from clipping to white.
    color = color / (vec3<f32>(1.0) + color);
    color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}
