//! Physical model: parameter derivation and the bubble nucleation schedule.
//!
//! Units are natural (`c = hbar = k_B = 1`) and lengths are measured in lattice
//! spacings.  Three choices fix the remaining freedom:
//!
//!   * `phi_b = 1` defines the unit of the scalar field,
//!   * `T_ref = 1` defines the unit of temperature, which fixes the radiation
//!     constant `a` through `e_ref = a T_ref^4`,
//!   * `dx = 1` defines the unit of length (and hence of time, since `c = 1`).
//!
//! Everything the user tunes is therefore dimensionless.

use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

/// The floating point type used for physical state on the host.
///
/// The solver is single precision throughout.  Widening it is a backend
/// question rather than a source-level one; see `docs/PRECISION.md`.
pub type Scalar = f32;

/// How large a bubble is when it is stamped into the lattice.
///
/// The physically meaningful scale is the critical radius `R_c`: a bubble
/// below it collapses under its own surface tension, a bubble above it grows.
/// A genuine thermal fluctuation nucleates *at* `R_c`, so the closer to 1 the
/// factor is, the more faithful the seeding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SeedSize {
    /// A multiple of the critical radius.  Tracks `R_c`, so the seeding stays
    /// physically meaningful when the potential is retuned live.
    Critical(Scalar),
    /// A fixed number of lattice cells, regardless of `R_c`.
    Cells(Scalar),
}

impl SeedSize {
    /// Just above critical: the smallest seed that reliably grows rather than
    /// collapsing.  See `Model::seed_radius` for why it is not exactly 1.
    pub const DEFAULT_FACTOR: Scalar = 1.15;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Model {
    /// Quartic coupling of the scalar potential.  Sets the wall thickness
    /// (`l_w = 2 sqrt(2) / (phi_b sqrt(lambda))`) and the surface tension.
    pub lambda: Scalar,
    /// Vacuum energy difference as a fraction of its maximum admissible value
    /// `lambda phi_b^4 / 12`.  At `1` the barrier disappears; small values give
    /// a strongly first-order transition with a large critical bubble.
    pub eps_ratio: Scalar,
    /// Transition strength `alpha = eps / e_ref`: latent heat relative to the
    /// radiation energy density of the plasma.
    pub alpha: Scalar,
    /// Phenomenological wall friction.  This is the one free knob that sets the
    /// terminal wall velocity, through the balance
    /// `eps ~ eta * sigma * gamma_w * v_w`.
    pub eta: Scalar,
    /// True-vacuum field value.  Fixed at 1 by the choice of field units.
    pub phi_b: Scalar,
    /// Lattice spacing.  Fixed at 1 by the choice of length units.
    pub dx: Scalar,
    /// Courant number: `dt = cfl * dx`.
    pub cfl: Scalar,
    /// Radius at which bubbles are stamped in.
    pub seed_size: SeedSize,
}

impl Default for Model {
    fn default() -> Self {
        Self {
            lambda: 0.5,
            eps_ratio: 0.3,
            alpha: 0.1,
            eta: 0.3,
            phi_b: 1.0,
            dx: 1.0,
            cfl: 0.2,
            seed_size: SeedSize::Critical(SeedSize::DEFAULT_FACTOR),
        }
    }
}

impl Model {
    /// Bag constant: the vacuum energy released per unit volume converted.
    ///
    /// Constrained to `eps < lambda phi_b^4 / 12`, which is exactly the
    /// condition for the symmetric minimum to survive (`V''(0) > 0`).
    pub fn eps(&self) -> Scalar {
        self.eps_ratio * self.lambda * self.phi_b.powi(4) / 12.0
    }

    /// `V''(0)`, the symmetric-phase curvature.
    pub fn m2(&self) -> Scalar {
        0.5 * self.lambda * self.phi_b.powi(2) - 6.0 * self.eps() / self.phi_b.powi(2)
    }

    /// Cubic coefficient of the potential (the barrier).
    pub fn delta(&self) -> Scalar {
        1.5 * self.lambda * self.phi_b - 6.0 * self.eps() / self.phi_b.powi(3)
    }

    /// Kink thickness of the planar wall, in the same units as `dx`.
    ///
    /// Exact for the degenerate potential (`eps -> 0`); the true wall is
    /// slightly thinner for `eps > 0`.  Keep this above ~3 cells or the wall
    /// will be under-resolved and will pick up lattice artefacts.
    pub fn wall_width(&self) -> Scalar {
        2.0 * std::f32::consts::SQRT_2 / (self.phi_b * self.lambda.sqrt())
    }

    /// Surface tension `sigma = integral sqrt(2V) dphi` of the degenerate wall.
    pub fn surface_tension(&self) -> Scalar {
        (self.lambda / 2.0).sqrt() * self.phi_b.powi(3) / 6.0
    }

    /// Thin-wall critical radius `R_c = 2 sigma / eps`.  Bubbles seeded smaller
    /// than this collapse instead of growing.
    pub fn critical_radius(&self) -> Scalar {
        2.0 * self.surface_tension() / self.eps()
    }

    /// Initial (symmetric-phase) fluid energy density.
    pub fn e_ref(&self) -> Scalar {
        self.eps() / self.alpha
    }

    pub fn t_ref(&self) -> Scalar {
        1.0
    }

    /// Radiation constant in `e = a T^4`, chosen so that `T_ref = 1`.
    pub fn a_rad(&self) -> Scalar {
        self.e_ref() / self.t_ref().powi(4)
    }

    pub fn dt(&self) -> Scalar {
        self.cfl * self.dx
    }

    /// Radius at which bubbles are stamped in, in the same units as `dx`.
    ///
    /// The default is `1.15 R_c` rather than exactly `R_c` for a specific
    /// reason: the critical bubble is in *unstable* equilibrium, so at exactly
    /// `R_c` the direction it moves is decided by discretisation error rather
    /// than physics, and it collapses about as often as it grows.  A few
    /// percent above critical is the smallest seed that reliably grows, and is
    /// far closer to a real thermal fluctuation than the comfortable margin a
    /// simulation is usually given.
    ///
    /// Seeding *below* `R_c` is allowed and does the physically correct thing:
    /// the bubble collapses.
    pub fn seed_radius(&self) -> Scalar {
        match self.seed_size {
            SeedSize::Critical(k) => k * self.critical_radius(),
            SeedSize::Cells(r) => r,
        }
        .max(0.25 * self.dx)
    }

    /// Ratio of critical radius to wall thickness.
    ///
    /// For this potential the thin-wall expressions collapse to the exact
    /// relation `R_c / l_w = 1 / eps_ratio`, independent of `lambda` and
    /// `phi_b`.  Two consequences worth knowing:
    ///
    ///   * the critical bubble is never *smaller* than its own wall, so there
    ///     is a hard floor on how small a physically meaningful seed can be;
    ///   * `eps_ratio` alone controls how well the thin-wall picture applies.
    ///     Below about 3 the bubble is a "thick-wall" one and `R_c = 2 sigma /
    ///     eps` is only indicative.
    pub fn critical_to_wall_ratio(&self) -> Scalar {
        1.0 / self.eps_ratio
    }

    /// Rough terminal wall velocity from the force balance
    /// `eps = eta * sigma * gamma_w * v_w`, ignoring the back-reaction of the
    /// fluid.  Shown in the UI as an orientation aid, not as a prediction.
    pub fn estimated_wall_velocity(&self) -> Scalar {
        let x = self.eps() / (self.eta.max(1e-6) * self.surface_tension());
        // Solve gamma v = x for v, i.e. v = x / sqrt(1 + x^2).
        x / (1.0 + x * x).sqrt()
    }

    /// Sound speed of the bag-model plasma.
    pub fn sound_speed(&self) -> Scalar {
        (1.0f32 / 3.0).sqrt()
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..1.0).contains(&self.eps_ratio) || self.eps_ratio <= 0.0 {
            return Err("eps_ratio must lie in (0, 1)".into());
        }
        if self.lambda <= 0.0 {
            return Err("lambda must be positive".into());
        }
        if self.alpha <= 0.0 {
            return Err("alpha must be positive".into());
        }
        if self.eta < 0.0 {
            return Err("eta must be non-negative".into());
        }
        if self.cfl <= 0.0 || self.cfl > 0.5 {
            return Err("cfl must lie in (0, 0.5]; see docs/NUMERICS.md".into());
        }
        match self.seed_size {
            SeedSize::Critical(k) if k <= 0.0 => {
                return Err("seed factor must be positive".into());
            }
            SeedSize::Cells(r) if r <= 0.0 => {
                return Err("seed radius must be positive".into());
            }
            _ => {}
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
//  Nucleation
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum NucleationMode {
    /// All bubbles appear at t = 0.  Every bubble then has the same size at any
    /// later time, which makes the geometry of the collision network easy to
    /// read.
    Simultaneous,
    /// Nucleation rate grows as exp(beta t), the physically relevant case:
    /// early bubbles grow large before the box fills, giving a broad
    /// distribution of bubble sizes.
    Exponential,
}

#[derive(Clone, Copy, Debug)]
pub struct BubbleEvent {
    pub time: Scalar,
    pub pos: [Scalar; 3],
    pub radius: Scalar,
}

/// Build the full list of nucleation events up front.
///
/// Positions are drawn uniformly at random subject to a minimum separation, so
/// that seeded bubbles do not start out already overlapping (which would look
/// like one deformed bubble rather than two colliding ones).
pub fn make_schedule(
    mode: NucleationMode,
    count: usize,
    radius: Scalar,
    grid: [u32; 3],
    duration: Scalar,
    seed: u64,
) -> Vec<BubbleEvent> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let n = [grid[0] as Scalar, grid[1] as Scalar, grid[2] as Scalar];
    let min_sep = 2.5 * radius;

    let mut events: Vec<BubbleEvent> = Vec::with_capacity(count);
    for k in 0..count {
        // --- time -----------------------------------------------------------
        let time = match mode {
            NucleationMode::Simultaneous => 0.0,
            NucleationMode::Exponential => {
                // Cumulative count n(t) proportional to exp(beta t) - 1, with
                // beta chosen so that the rate grows by e^4 over `duration`.
                let growth = 4.0f32;
                let frac = (k as Scalar + 0.5) / count as Scalar;
                duration / growth * (1.0 + frac * (growth.exp() - 1.0)).ln()
            }
        };

        // --- position -------------------------------------------------------
        let mut pos = [0.0; 3];
        for _attempt in 0..64 {
            let candidate = [
                rng.random::<Scalar>() * n[0],
                rng.random::<Scalar>() * n[1],
                rng.random::<Scalar>() * n[2],
            ];
            let ok = events.iter().all(|e| {
                periodic_distance(candidate, e.pos, n) > min_sep
            });
            pos = candidate;
            if ok {
                break;
            }
        }

        events.push(BubbleEvent { time, pos, radius });
    }

    events.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
    events
}

fn periodic_distance(a: [Scalar; 3], b: [Scalar; 3], n: [Scalar; 3]) -> Scalar {
    let mut sum = 0.0;
    for i in 0..3 {
        let mut d = (a[i] - b[i]).abs();
        if d > 0.5 * n[i] {
            d = n[i] - d;
        }
        sum += d * d;
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The derived coefficients must reproduce the three conditions the
    /// potential was constructed to satisfy.
    #[test]
    fn potential_has_the_requested_shape() {
        let m = Model::default();
        let (m2, delta, lambda, phi_b) = (m.m2(), m.delta(), m.lambda, m.phi_b);
        let v = |p: f32| 0.5 * m2 * p * p - delta * p * p * p / 3.0 + 0.25 * lambda * p.powi(4);
        let dv = |p: f32| p * (m2 - delta * p + lambda * p * p);

        assert!(m2 > 0.0, "symmetric minimum must be metastable");
        assert!(dv(phi_b).abs() < 1e-6, "phi_b must be a stationary point");
        assert!(
            (v(phi_b) + m.eps()).abs() < 1e-6,
            "V(phi_b) must equal -eps, got {} vs {}",
            v(phi_b),
            -m.eps()
        );
        // And there must be a barrier in between.
        let top = (0..100)
            .map(|i| v(phi_b * i as f32 / 100.0))
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(top > 0.0, "a first-order transition needs a barrier");
    }

    #[test]
    fn degenerate_limit_is_the_symmetric_double_well() {
        let m = Model { eps_ratio: 1e-6, ..Model::default() };
        // V = (lambda/4) phi^2 (phi - phi_b)^2 has m2 = lambda/2, delta = 3 lambda/2.
        assert!((m.m2() - 0.5 * m.lambda).abs() < 1e-4);
        assert!((m.delta() - 1.5 * m.lambda).abs() < 1e-4);
    }

    /// The thin-wall expressions collapse to R_c / l_w = 1 / eps_ratio exactly.
    /// This is what puts a floor under how small a meaningful seed can be.
    #[test]
    fn critical_radius_is_never_smaller_than_the_wall() {
        for eps_ratio in [0.05, 0.2, 0.3, 0.6, 0.95] {
            for lambda in [0.1, 0.5, 1.5] {
                let m = Model { eps_ratio, lambda, ..Model::default() };
                let ratio = m.critical_radius() / m.wall_width();
                assert!(
                    (ratio - m.critical_to_wall_ratio()).abs() < 1e-3 * ratio,
                    "R_c/l_w should be 1/eps_ratio: got {ratio} vs {}",
                    m.critical_to_wall_ratio()
                );
                assert!(ratio > 1.0, "the critical bubble must exceed its own wall");
            }
        }
    }

    #[test]
    fn seed_radius_follows_the_chosen_mode() {
        let m = Model { seed_size: SeedSize::Critical(1.15), ..Model::default() };
        assert!((m.seed_radius() - 1.15 * m.critical_radius()).abs() < 1e-4);

        // A factor mode tracks R_c when the potential is retuned.
        let softer = Model { eps_ratio: 0.6, ..m };
        assert!(softer.seed_radius() < m.seed_radius());

        // A cells mode does not.
        let fixed = Model { seed_size: SeedSize::Cells(7.5), ..Model::default() };
        assert!((fixed.seed_radius() - 7.5).abs() < 1e-6);
        let fixed_softer = Model { eps_ratio: 0.6, ..fixed };
        assert!((fixed_softer.seed_radius() - 7.5).abs() < 1e-6);
    }

    #[test]
    fn sub_critical_seeds_are_allowed_so_collapse_can_be_shown() {
        let m = Model { seed_size: SeedSize::Critical(0.5), ..Model::default() };
        m.validate().expect("sub-critical seeding is physical, not an error");
        assert!(m.seed_radius() < m.critical_radius());
    }

    #[test]
    fn schedule_respects_ordering_and_bounds() {
        let grid = [64, 64, 64];
        let ev = make_schedule(NucleationMode::Exponential, 12, 8.0, grid, 40.0, 7);
        assert_eq!(ev.len(), 12);
        for w in ev.windows(2) {
            assert!(w[0].time <= w[1].time);
        }
        for e in &ev {
            assert!(e.time >= 0.0 && e.time <= 40.0 + 1e-3);
            for (axis, n) in grid.iter().enumerate() {
                assert!(e.pos[axis] >= 0.0 && e.pos[axis] <= *n as f32);
            }
        }
    }
}
