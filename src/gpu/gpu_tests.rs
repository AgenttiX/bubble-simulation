//! Tests that need a real device.
//!
//! Marked `#[ignore]` so `cargo test` stays runnable without a GPU; run them
//! with `cargo test -- --ignored`.
//!
//! These cover the lattice resize path, which cannot be exercised from the
//! ordinary headless run: it reallocates every buffer, rebuilds every bind
//! group, and re-points the renderer at a new visualisation texture, and a
//! mistake in any of that shows up as a device loss rather than a wrong number.

use crate::gpu::preprocess::Precision;
use crate::gpu::render::{RenderSettings, VolumeRenderer};
use crate::gpu::sim::{Simulation, SimulationSpec};
use crate::physics::{Model, NucleationMode};

fn test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("test"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .ok()
    })
}

fn spec(grid: [u32; 3]) -> SimulationSpec {
    SimulationSpec {
        model: Model::default(),
        grid,
        nucleation: NucleationMode::Simultaneous,
        bubbles: 1,
        nucleation_duration: 10.0,
        seed: 1,
        precision: Precision::Single,
    }
}

/// Advance a few steps and confirm the state is still physical.
///
/// A resize that leaves a stale bind group pointing at a freed buffer, or a
/// grid mismatch between the uniform and the dispatch, shows up here as NaNs
/// or as a broken-phase fraction outside [0, 1].
fn step_and_check(device: &wgpu::Device, queue: &wgpu::Queue, sim: &mut Simulation, what: &str) {
    for _ in 0..8 {
        let mut enc = device.create_command_encoder(&Default::default());
        sim.record_steps(queue, &mut enc, 1);
        queue.submit(Some(enc.finish()));
    }
    let mut enc = device.create_command_encoder(&Default::default());
    sim.record_visualisation(&mut enc);
    sim.record_diagnostics(&mut enc);
    queue.submit(Some(enc.finish()));
    sim.after_submit();
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let d = sim.poll_diagnostics();
    assert!(d.mean_energy.is_finite(), "{what}: energy is not finite");
    assert!(d.max_velocity.is_finite(), "{what}: max |v| is not finite");
    assert!(d.mean_energy > 0.0, "{what}: energy collapsed to {}", d.mean_energy);
    assert!(
        (0.0..=1.0).contains(&d.broken_fraction),
        "{what}: broken fraction {} is outside [0, 1]",
        d.broken_fraction
    );
    assert!(d.max_velocity < 1.0, "{what}: max |v| = {} is superluminal", d.max_velocity);
}

#[test]
#[ignore = "requires a GPU"]
fn resize_rebuilds_a_working_simulation() {
    let Some((device, queue)) = test_device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };

    let mut sim = Simulation::new(&device, &queue, spec([64; 3])).unwrap();
    assert_eq!(sim.cell_count(), 64 * 64 * 64);
    step_and_check(&device, &queue, &mut sim, "initial 64^3");

    // Grow.
    sim.resize(&device, &queue, [96; 3]).unwrap();
    assert_eq!(sim.grid, [96; 3]);
    assert_eq!(sim.cell_count(), 96 * 96 * 96);
    assert_eq!(sim.time, 0.0, "a resize restarts the run");
    step_and_check(&device, &queue, &mut sim, "grown to 96^3");

    // Shrink. This is the case that would fail if the old lattice were still
    // held while the new one is allocated on a nearly full device.
    sim.resize(&device, &queue, [32; 3]).unwrap();
    assert_eq!(sim.cell_count(), 32 * 32 * 32);
    step_and_check(&device, &queue, &mut sim, "shrunk to 32^3");

    // Non-cubic, which exercises the per-axis indexing rather than assuming
    // nx == ny == nz anywhere.
    sim.resize(&device, &queue, [48, 32, 72]).unwrap();
    assert_eq!(sim.grid, [48, 32, 72]);
    assert_eq!(sim.cell_count(), 48 * 32 * 72);
    step_and_check(&device, &queue, &mut sim, "non-cubic 48x32x72");
}

/// The old lattice must actually be released, not merely dropped and leaked.
///
/// `wgpu` defers destruction until no queued submission still references a
/// resource, so a resize that forgot to drain the queue would keep the old
/// buffers alive and double peak memory. Its own allocator report is the
/// direct way to check.
#[test]
#[ignore = "requires a GPU"]
fn resize_releases_the_old_lattice() {
    let Some((device, queue)) = test_device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let allocated = |device: &wgpu::Device| {
        device
            .generate_allocator_report()
            .map(|r| r.total_allocated_bytes)
    };
    let Some(baseline) = allocated(&device) else {
        eprintln!("backend does not report allocations; skipping");
        return;
    };

    let big = [256u32; 3];
    let small = [64u32; 3];
    let big_bytes = Simulation::lattice_bytes(big);

    let mut sim = Simulation::new(&device, &queue, spec(big)).unwrap();
    step_and_check(&device, &queue, &mut sim, "big");
    let peak = allocated(&device).unwrap();
    assert!(
        peak >= baseline + big_bytes / 2,
        "expected the {:.2} GB lattice to show up in the allocator report",
        big_bytes as f64 / 1e9
    );

    sim.resize(&device, &queue, small).unwrap();
    let after = allocated(&device).unwrap();

    // Nearly all of the big lattice should be gone; allow generous slack for
    // the small replacement and wgpu's own bookkeeping.
    let released = peak.saturating_sub(after);
    assert!(
        released > (big_bytes * 3) / 4,
        "resize released only {:.2} GB of a {:.2} GB lattice",
        released as f64 / 1e9,
        big_bytes as f64 / 1e9,
    );
    step_and_check(&device, &queue, &mut sim, "after release");
}

/// The predicted size must match what is actually allocated, since the memory
/// cap in the UI is computed from the prediction.
#[test]
#[ignore = "requires a GPU"]
fn predicted_lattice_size_matches_the_allocation() {
    let Some((device, queue)) = test_device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    for grid in [[64u32; 3], [128, 64, 96], [192; 3]] {
        let sim = Simulation::new(&device, &queue, spec(grid)).unwrap();
        let predicted = Simulation::lattice_bytes(grid) as f64;
        let actual = sim.device_memory_bytes() as f64;
        let error = (actual - predicted).abs() / predicted;
        assert!(
            error < 0.01,
            "{grid:?}: predicted {predicted:.0} B but allocated {actual:.0} B ({:.2}% off)",
            error * 100.0
        );
    }
}

/// A rejected resize must leave the running simulation untouched -- which is
/// why the size is validated before anything is released.
#[test]
#[ignore = "requires a GPU"]
fn a_rejected_resize_leaves_the_simulation_intact() {
    let Some((device, queue)) = test_device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };

    let mut sim = Simulation::new(&device, &queue, spec([64; 3])).unwrap();

    // Below the stencil minimum.
    assert!(sim.resize(&device, &queue, [4; 3]).is_err());
    assert_eq!(sim.grid, [64; 3], "a rejected resize must not change the grid");

    // Beyond the storage-binding limit.
    let limits = device.limits();
    let too_big = Simulation::max_cubic_size(limits.max_storage_buffer_binding_size, None).side + 64;
    assert!(sim.resize(&device, &queue, [too_big; 3]).is_err());
    assert_eq!(sim.grid, [64; 3]);

    step_and_check(&device, &queue, &mut sim, "after rejected resizes");
}

/// The renderer must survive being re-pointed at a new visualisation texture.
#[test]
#[ignore = "requires a GPU"]
fn renderer_rebinds_after_a_resize() {
    let Some((device, queue)) = test_device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };

    let mut sim = Simulation::new(&device, &queue, spec([64; 3])).unwrap();
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let mut renderer = VolumeRenderer::new(&device, format, &sim.vis_view, sim.grid);
    let before = renderer.box_half();

    sim.resize(&device, &queue, [96, 48, 48]).unwrap();
    renderer.rebind(&device, &sim.vis_view, sim.grid);

    // A non-cubic lattice must produce a non-cubic box.
    let after = renderer.box_half();
    assert_ne!(before, after);
    assert!(after.x > after.y, "the long axis should have the largest half-extent");

    // Draw a frame against the new texture; a stale bind group would trip
    // validation here.
    let out = std::env::temp_dir().join("bubble-sim-rebind-test.png");
    crate::capture::render_to_png(
        &device,
        &queue,
        &sim.vis_view,
        sim.grid,
        &RenderSettings::default(),
        (64, 64),
        &out,
    )
    .expect("render after rebind");
    assert!(std::fs::metadata(&out).is_ok_and(|m| m.len() > 0));
    let _ = std::fs::remove_file(&out);
}
