//! Windowless run: advance the lattice and print diagnostics.
//!
//! Two uses.  It benchmarks the solver without the renderer competing for the
//! GPU, and it is how the physics is checked: the transition should complete,
//! the plasma should be stirred, and total energy should be conserved once
//! nucleation is over.  None of that is visible from a screenshot.

use anyhow::{Context, Result};

use crate::capture;
use crate::config::Config;
use crate::gpu::render::RenderSettings;
use crate::gpu::sim::{Diagnostics, Simulation};

pub struct Report {
    pub time: f32,
    pub diag: Diagnostics,
    /// Radius of a single sphere with the same volume as the broken phase, in
    /// lattice cells.  Meaningful while bubbles are still separate.
    pub equivalent_radius: f32,
}

pub fn run(config: &Config, steps: u64, report_every: u64) -> Result<()> {
    let (device, queue) = pollster::block_on(headless_device())?;
    let model = config.model();
    let grid = config.grid();
    let mut simulation = Simulation::new(&device, &queue, config.sim_spec())
        .context("failed to create the simulation")?;

    println!(
        "lattice {}x{}x{} = {:.2} M cells | {:.2} GB | dt = {:.3}",
        grid[0],
        grid[1],
        grid[2],
        simulation.cell_count() as f64 / 1e6,
        simulation.device_memory_bytes() as f64 / 1e9,
        model.dt(),
    );
    println!(
        "wall {:.2} cells | R_c {:.1} cells | sigma {:.4} | eps {:.4} | e_ref {:.4}",
        model.wall_width(),
        model.critical_radius(),
        model.surface_tension(),
        model.eps(),
        model.e_ref(),
    );
    if steps > 0 {
        println!(
            "\n{:>10} {:>10} {:>10} {:>10} {:>11} {:>11} {:>10}",
            "step", "time", "broken", "R_eq", "max|v|", "K", "E/E0"
        );
    }

    let mut baseline: Option<f32> = None;
    let mut done = 0u64;
    let start = std::time::Instant::now();

    while done < steps {
        let chunk = report_every.min(steps - done);
        let report = advance(&device, &queue, &mut simulation, chunk)?;
        done += chunk;

        anyhow::ensure!(
            report.diag.mean_energy.is_finite() && report.diag.max_velocity.is_finite(),
            "the solution blew up at step {done} (non-finite diagnostics); \
             reduce --cfl or --alpha"
        );

        // Baseline the conservation check only once every bubble is in, since
        // stamping a bubble adds its energy to the box by hand.
        if baseline.is_none() && simulation.remaining_bubbles() == 0 {
            baseline = Some(report.diag.mean_energy);
        }
        // Until every bubble is in there is nothing to compare against.
        let ratio = match baseline {
            Some(b) => format!("{:.6}", report.diag.mean_energy / b),
            None => "-".to_string(),
        };

        println!(
            "{:>10} {:>10.1} {:>9.2}% {:>10.1} {:>11.4} {:>11.3e} {:>10}",
            done,
            report.time,
            100.0 * report.diag.broken_fraction,
            report.equivalent_radius,
            report.diag.max_velocity,
            report.diag.kinetic_fraction(),
            ratio,
        );
    }

    let elapsed = start.elapsed().as_secs_f64();
    if steps > 0 {
        let cell_steps = steps as f64 * simulation.cell_count() as f64;
        println!(
            "\n{steps} steps in {elapsed:.2} s  ({:.1} steps/s, {:.2} G cell-updates/s)",
            steps as f64 / elapsed,
            cell_steps / elapsed / 1e9,
        );
    }

    if let Some(path) = &config.screenshot {
        // The visualisation texture already holds the current state: `advance`
        // refreshes it, and `Simulation::new` does so for step 0.
        capture::render_to_png(
            &device,
            &queue,
            &simulation.vis_view,
            grid,
            &RenderSettings::with_field(config.field),
            config.screenshot_size,
            path,
        )?;
    }
    Ok(())
}

/// Advance `n` steps and block until the diagnostics for the resulting state
/// are available.  Blocking is fine here: there is no frame to keep up with.
fn advance(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    simulation: &mut Simulation,
    n: u64,
) -> Result<Report> {
    for _ in 0..n {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("step") });
        simulation.record_steps(queue, &mut encoder, 1);
        queue.submit(Some(encoder.finish()));
    }

    let mut encoder = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("diagnostics") });
    // `record_diagnostics` reads the primitive cache, which this refreshes.
    simulation.record_visualisation(&mut encoder);
    simulation.record_diagnostics(&mut encoder);
    queue.submit(Some(encoder.finish()));
    simulation.after_submit();
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .context("device poll failed")?;

    let diag = simulation.poll_diagnostics();
    let cells = simulation.cell_count() as f32;
    let volume = diag.broken_fraction * cells;
    let equivalent_radius = (volume * 3.0 / (4.0 * std::f32::consts::PI)).cbrt();

    Ok(Report { time: simulation.time, diag, equivalent_radius })
}

async fn headless_device() -> Result<(wgpu::Device, wgpu::Queue)> {
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
        .context("no suitable GPU adapter found")?;
    let info = adapter.get_info();
    println!("adapter: {} ({:?})", info.name, info.backend);

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("bubble simulation (headless)"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await
        .context("failed to acquire a device")?;
    device.on_uncaptured_error(std::sync::Arc::new(|err| {
        panic!("wgpu error: {err}");
    }));
    Ok((device, queue))
}
