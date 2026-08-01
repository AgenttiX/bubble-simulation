//! Live GPU lattice simulation of bubble nucleation and growth in a
//! first-order cosmological phase transition.
//!
//! See `docs/PHYSICS.md` for the model, `docs/NUMERICS.md` for the scheme,
//! `docs/EOS.md` for how to replace the bag equation of state, and
//! `docs/PRECISION.md` for the single- to double-precision story.

mod app;
mod camera;
mod capture;
mod config;
mod gpu;
mod headless;
mod physics;
mod ui;

use anyhow::Result;
use clap::Parser;
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("bubble_simulation=info,wgpu_core=warn"),
    )
    .init();

    let config = config::Config::parse();
    config.validate()?;

    if config.headless.is_some() || config.screenshot.is_some() {
        let steps = config.headless.unwrap_or(0);
        let every = config.report_every.unwrap_or((steps / 20).max(1));
        return headless::run(&config, steps, every);
    }

    let event_loop = EventLoop::new()?;
    // Free-running: the simulation advances as fast as the GPU allows, with
    // presentation paced by the swapchain's present mode.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = app::App::new(config);
    event_loop.run_app(&mut app)?;
    Ok(())
}
