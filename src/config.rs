//! Command line configuration.

use anyhow::{Result, bail};
use clap::Parser;

use crate::gpu::preprocess::Precision;
use crate::gpu::render::FieldMode;
use crate::gpu::sim::SimulationSpec;
use crate::physics::{Model, NucleationMode, SeedSize};

/// Live GPU lattice simulation of bubble nucleation and growth in a
/// first-order cosmological phase transition.
#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
pub struct Config {
    /// Lattice size. Either one number for a cubic box (`--grid 192`) or three
    /// comma-separated numbers (`--grid 256,128,128`).
    ///
    /// Memory is about 88 bytes per cell (three RK state slots plus the
    /// primitive cache): 192^3 needs 0.62 GB, 256^3 1.48 GB, 384^3 5.0 GB.
    #[arg(long, default_value = "192", value_parser = parse_grid)]
    pub grid: [u32; 3],

    /// Quartic coupling of the scalar potential. Sets the wall thickness
    /// `l_w = 2 sqrt(2) / sqrt(lambda)` in lattice units; keep `l_w` above ~3.
    #[arg(long, default_value_t = 0.5)]
    pub lambda: f32,

    /// Vacuum energy split, as a fraction of its maximum `lambda phi_b^4 / 12`.
    /// Small values give a tall barrier and a large critical bubble.
    #[arg(long, default_value_t = 0.3)]
    pub eps_ratio: f32,

    /// Transition strength: latent heat over plasma radiation energy density.
    #[arg(long, default_value_t = 0.1)]
    pub alpha: f32,

    /// Wall friction. Large values give slow deflagrations with a shock ahead
    /// of the wall; small values give fast detonations.
    #[arg(long, default_value_t = 0.3)]
    pub eta: f32,

    /// Courant number, `dt = cfl * dx`. The hard limit is 0.5, set by the
    /// scalar field's wave equation under SSP-RK3; accuracy degrades long
    /// before that, so 0.2 is the recommended value.
    #[arg(long, default_value_t = 0.2)]
    pub cfl: f32,

    /// Radius at which bubbles are stamped in, as a multiple of the critical
    /// radius R_c. A real thermal fluctuation nucleates at exactly R_c, so
    /// values just above 1 are the physically faithful choice; below 1 the
    /// bubble collapses, which is also correct and worth watching.
    ///
    /// Exactly 1.0 is unstable equilibrium: discretisation error decides
    /// whether it grows or collapses.
    #[arg(long, default_value_t = SeedSize::DEFAULT_FACTOR)]
    pub seed_factor: f32,

    /// Seed bubbles at a fixed radius in lattice cells instead, ignoring
    /// `--seed-factor`. Useful for holding the seed fixed while sweeping the
    /// potential.
    #[arg(long, value_name = "CELLS")]
    pub seed_radius: Option<f32>,

    /// Number of bubbles to seed over the course of the run.
    #[arg(long, default_value_t = 12)]
    pub bubbles: usize,

    /// How bubbles are distributed in time.
    #[arg(long, value_enum, default_value_t = NucleationMode::Exponential)]
    pub nucleation: NucleationMode,

    /// Time over which exponential nucleation completes.
    #[arg(long, default_value_t = 60.0)]
    pub nucleation_duration: f32,

    /// Seed for bubble placement.
    #[arg(long, default_value_t = 1)]
    pub seed: u64,

    /// Time steps advanced per rendered frame.
    #[arg(long, default_value_t = 2)]
    pub steps_per_frame: u32,

    /// Which plasma field the volume renderer starts on. All four remain
    /// selectable in the panel.
    #[arg(long, value_enum, default_value_t = FieldMode::Kinetic)]
    pub field: FieldMode,

    /// Present frames as fast as the GPU can produce them instead of waiting
    /// for vertical blank.
    #[arg(long)]
    pub no_vsync: bool,

    /// Run this many steps without opening a window, printing diagnostics, then
    /// exit. Useful for benchmarking and for checking that the transition
    /// completes and energy is conserved.
    #[arg(long)]
    pub headless: Option<u64>,

    /// How often to report in headless mode. Defaults to 20 evenly spaced
    /// reports over the run.
    #[arg(long)]
    pub report_every: Option<u64>,

    /// Render one frame to this PNG and exit. Implies a windowless run, so
    /// combine with `--headless N` to choose how far the simulation has
    /// advanced when the frame is taken.
    #[arg(long, value_name = "PATH")]
    pub screenshot: Option<std::path::PathBuf>,

    /// Resolution of `--screenshot`, as WIDTHxHEIGHT.
    #[arg(long, default_value = "1280x960", value_parser = parse_size)]
    pub screenshot_size: (u32, u32),
}

impl Config {
    pub fn grid(&self) -> [u32; 3] {
        self.grid
    }

    pub fn model(&self) -> Model {
        Model {
            lambda: self.lambda,
            eps_ratio: self.eps_ratio,
            alpha: self.alpha,
            eta: self.eta,
            phi_b: 1.0,
            dx: 1.0,
            cfl: self.cfl,
            seed_size: match self.seed_radius {
                Some(cells) => SeedSize::Cells(cells),
                None => SeedSize::Critical(self.seed_factor),
            },
        }
    }

    pub fn sim_spec(&self) -> SimulationSpec {
        SimulationSpec {
            model: self.model(),
            grid: self.grid,
            nucleation: self.nucleation,
            bubbles: self.bubbles,
            nucleation_duration: self.nucleation_duration,
            seed: self.seed,
            // WGSL has no f64; see docs/PRECISION.md.
            precision: Precision::Single,
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.model().validate().map_err(anyhow::Error::msg)?;
        if self.bubbles == 0 {
            bail!("--bubbles must be at least 1");
        }
        if self.steps_per_frame == 0 {
            bail!("--steps-per-frame must be at least 1");
        }
        if self.nucleation_duration <= 0.0 {
            bail!("--nucleation-duration must be positive");
        }
        let m = self.model();
        if m.wall_width() < 2.0 {
            bail!(
                "lambda = {} gives a wall only {:.2} cells thick, which the lattice cannot \
                 resolve; use lambda <= {:.2}",
                self.lambda,
                m.wall_width(),
                2.0f32,
            );
        }
        for (axis, n) in self.grid.iter().enumerate() {
            if *n < 8 {
                bail!("grid axis {axis} is {n}; the 5-point stencil needs at least 8 cells");
            }
        }
        Ok(())
    }
}

fn parse_size(s: &str) -> Result<(u32, u32), String> {
    let (w, h) = s
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("`{s}` is not a size; expected WIDTHxHEIGHT"))?;
    let parse = |v: &str, what: &str| {
        v.trim()
            .parse::<u32>()
            .map_err(|e| format!("{what}: {e}"))
            .and_then(|n| (n > 0).then_some(n).ok_or_else(|| format!("{what} must be positive")))
    };
    Ok((parse(w, "width")?, parse(h, "height")?))
}

fn parse_grid(s: &str) -> Result<[u32; 3], String> {
    let parts: Vec<&str> = s.split(',').map(str::trim).collect();
    let nums: Result<Vec<u32>, _> = parts.iter().map(|p| p.parse::<u32>()).collect();
    let nums = nums.map_err(|e| format!("`{s}` is not a lattice size: {e}"))?;
    match nums.len() {
        1 => Ok([nums[0]; 3]),
        3 => Ok([nums[0], nums[1], nums[2]]),
        n => Err(format!("expected 1 or 3 comma-separated sizes, got {n}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_parsing() {
        assert_eq!(parse_size("1920x1080").unwrap(), (1920, 1080));
        assert_eq!(parse_size("800X600").unwrap(), (800, 600));
        assert!(parse_size("1920").is_err());
        assert!(parse_size("0x600").is_err());
    }

    #[test]
    fn grid_accepts_one_or_three_values() {
        assert_eq!(parse_grid("192").unwrap(), [192, 192, 192]);
        assert_eq!(parse_grid("256,128, 64").unwrap(), [256, 128, 64]);
        assert!(parse_grid("1,2").is_err());
        assert!(parse_grid("abc").is_err());
    }

    #[test]
    fn defaults_are_valid_and_well_resolved() {
        let c = Config::parse_from(["bubble-simulation"]);
        c.validate().unwrap();
        let m = c.model();
        assert!(m.wall_width() >= 3.0, "wall is {} cells", m.wall_width());
        assert!(
            m.critical_radius() > m.wall_width(),
            "the thin-wall picture needs R_c > l_w"
        );
    }

    #[test]
    fn an_unresolvable_wall_is_rejected() {
        let mut c = Config::parse_from(["bubble-simulation"]);
        c.lambda = 50.0;
        assert!(c.validate().is_err());
    }
}
