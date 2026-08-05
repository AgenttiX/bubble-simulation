//! Live control panel.
//!
//! Everything here is either a solver parameter that is safe to change mid-run
//! (the shaders read them from a uniform, so the next dispatch simply uses new
//! constants) or a rendering choice.  Parameters that change the potential also
//! change the energy of the current configuration, which is called out in the
//! panel rather than hidden.

use crate::gpu::render::{COLORMAPS, FieldMode, RenderSettings};
use crate::gpu::sim::{self, Diagnostics};
use crate::gpu::vram::{self, VramInfo};
use crate::physics::{Model, NucleationMode, SeedSize};

pub struct UiState {
    pub visible: bool,
    pub running: bool,
    pub steps_per_frame: u32,
    pub nucleation_mode: NucleationMode,
    pub bubble_count: usize,
    pub nucleation_duration: f32,
    pub seed: u64,
    /// Lattice size being edited. Applied only on request, since changing it
    /// reallocates every buffer and restarts the run.
    pub lattice: [u32; 3],
    /// Whether the three axes are edited together.
    pub lattice_cubic: bool,
    /// Total energy at the moment the last nucleation finished; the drift
    /// readout is measured against this.
    pub energy_baseline: Option<f32>,
    pub fps: f32,
    /// Mean wall-clock time per frame over the last 64 frames, in milliseconds.
    /// Wall clock, not GPU time: with vsync on it is pinned to the display's
    /// refresh interval rather than measuring how much work the frame did.
    pub frame_ms: f32,
    /// Worst frame in the same window, which is where stutter shows up.
    pub frame_ms_max: f32,
    /// Whether presentation is waiting for vertical blank.
    pub vsync: bool,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            visible: true,
            running: true,
            steps_per_frame: 2,
            nucleation_mode: NucleationMode::Exponential,
            bubble_count: 12,
            nucleation_duration: 60.0,
            seed: 1,
            lattice: [192; 3],
            lattice_cubic: true,
            energy_baseline: None,
            fps: 0.0,
            frame_ms: 0.0,
            frame_ms_max: 0.0,
            vsync: true,
        }
    }
}

impl UiState {
    /// Point the lattice selection at `grid`.
    ///
    /// Also syncs the `cubic` toggle, because the cubic editor writes
    /// `[n; 3]` back every frame -- so leaving it set for a non-cubic grid
    /// would silently rewrite the selection and make the Apply button look
    /// live when the user has changed nothing.
    pub fn select_lattice(&mut self, grid: [u32; 3]) {
        self.lattice = grid;
        self.lattice_cubic = grid[0] == grid[1] && grid[1] == grid[2];
    }
}

/// What the user asked for this frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct UiCommands {
    pub reset: bool,
    pub reseed_and_reset: bool,
    pub nucleate_now: bool,
    pub single_step: bool,
    pub frame_view: bool,
    /// Rebuild at this lattice size.
    pub resize: Option<[u32; 3]>,
}

pub struct SimInfo {
    pub grid: [u32; 3],
    pub time: f32,
    pub steps: u64,
    pub cells: u64,
    pub bubbles_remaining: usize,
    pub device_bytes: u64,
    /// Device memory, when the backend can report it.
    pub vram: Option<VramInfo>,
    /// `max_storage_buffer_binding_size`, a hard API ceiling on the fluid
    /// buffer independent of how much memory is free.
    pub max_storage_binding: u64,
}

/// Draw the panel, recording whatever the user asked for into `cmd`.
///
/// `cmd` is borrowed rather than returned so that there is exactly one
/// `UiCommands` value in play, shared with the keyboard handler. An earlier
/// version returned a fresh one and merged it field by field at the call site,
/// which silently dropped any field the merge forgot -- as it did for
/// `resize`, leaving the Apply button doing nothing at all.
pub fn draw(
    root: &mut egui::Ui,
    state: &mut UiState,
    model: &mut Model,
    render: &mut RenderSettings,
    info: &SimInfo,
    diag: &Diagnostics,
    cmd: &mut UiCommands,
) {
    if !state.visible {
        return;
    }

    egui::Panel::left("controls")
        .resizable(true)
        .default_size(340.0)
        .show(root, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                transport(ui, state, cmd);
                ui.separator();
                diagnostics(ui, state, info, diag, model);
                ui.separator();
                physics(ui, model);
                ui.separator();
                lattice(ui, state, info, cmd);
                ui.separator();
                nucleation(ui, state, model, cmd);
                ui.separator();
                visualisation(ui, render, cmd);
                ui.separator();
                help(ui);
            });
        });
}

fn transport(ui: &mut egui::Ui, state: &mut UiState, cmd: &mut UiCommands) {
    ui.heading("Bubble nucleation");
    ui.horizontal(|ui| {
        let label = if state.running { "Pause" } else { "Run" };
        if ui.button(label).clicked() {
            state.running = !state.running;
        }
        if ui.button("Step").clicked() {
            cmd.single_step = true;
        }
        if ui.button("Reset").clicked() {
            cmd.reset = true;
        }
        if ui.button("New seed").clicked() {
            cmd.reseed_and_reset = true;
        }
    });
    ui.add(
        egui::Slider::new(&mut state.steps_per_frame, 1..=16)
            .text("steps / frame"),
    );
    frame_timing(ui, state);
}

/// Frame time, as the headline number rather than a footnote.
///
/// Wall-clock milliseconds per frame is the thing to watch when raising the
/// lattice size or the step count, so it gets the largest text in the panel.
/// The vsync caveat is spelled out because otherwise the reading is dominated
/// by the display's refresh rate and looks suspiciously constant.
fn frame_timing(ui: &mut egui::Ui, state: &UiState) {
    let response = ui
        .horizontal(|ui| {
            ui.label(
                egui::RichText::new(format!("{:.1}", state.frame_ms))
                    .monospace()
                    .strong()
                    .size(19.0),
            );
            ui.label(egui::RichText::new("ms / frame").size(13.0));
        })
        .response;

    // The per-step figure only means anything while steps are actually being
    // taken; while paused the frame time is pure rendering.
    let per_step = if state.running {
        format!(
            "   \u{2022}   {:.2} ms / step",
            state.frame_ms / state.steps_per_frame.max(1) as f32
        )
    } else {
        String::from("   \u{2022}   paused")
    };
    ui.label(
        egui::RichText::new(format!(
            "{:.0} fps   \u{2022}   worst {:.1} ms{per_step}",
            state.fps, state.frame_ms_max,
        ))
        .small()
        .weak(),
    );

    if state.vsync {
        ui.label(
            egui::RichText::new(
                "vsync on: frame time is capped by the display, so this is an upper \
                 bound on how fast the simulation could run. Start with --no-vsync to \
                 measure the real cost, or use --headless for a clean benchmark.",
            )
            .small()
            .weak(),
        );
    }

    response.on_hover_text(
        "Mean wall-clock time per frame over the last 64 frames. Covers the whole \
         frame: the simulation steps, the visualisation pass, the ray march and the \
         panel itself.",
    );
}

fn diagnostics(
    ui: &mut egui::Ui,
    state: &mut UiState,
    info: &SimInfo,
    diag: &Diagnostics,
    model: &Model,
) {
    egui::CollapsingHeader::new("Diagnostics").default_open(true).show(ui, |ui| {
        egui::Grid::new("diag").num_columns(2).striped(true).show(ui, |ui| {
            ui.label("lattice");
            ui.label(format!(
                "{}x{}x{}  ({:.1} M cells)",
                info.grid[0],
                info.grid[1],
                info.grid[2],
                info.cells as f64 / 1e6
            ));
            ui.end_row();

            ui.label("device memory");
            ui.label(format!("{:.2} GB", info.device_bytes as f64 / 1e9));
            ui.end_row();

            ui.label("time");
            ui.label(format!("{:.1}  ({} steps)", info.time, info.steps));
            ui.end_row();

            ui.label("bubbles pending");
            ui.label(format!("{}", info.bubbles_remaining));
            ui.end_row();

            ui.label("broken phase");
            ui.label(format!("{:.1} %", 100.0 * diag.broken_fraction));
            ui.end_row();

            ui.label("max |v|");
            ui.label(format!("{:.3} c", diag.max_velocity));
            ui.end_row();

            ui.label("kinetic fraction K");
            ui.label(format!("{:.3e}", diag.kinetic_fraction()));
            ui.end_row();

            ui.label("mean fluid energy");
            ui.label(format!("{:.4}  (initial {:.4})", diag.mean_fluid_energy, model.e_ref()));
            ui.end_row();

            ui.label("energy drift");
            match state.energy_baseline {
                Some(base) if base.abs() > 1e-20 => {
                    let drift = (diag.mean_energy - base) / base;
                    let colour = if drift.abs() < 1e-3 {
                        egui::Color32::from_rgb(120, 200, 120)
                    } else if drift.abs() < 1e-2 {
                        egui::Color32::from_rgb(220, 200, 110)
                    } else {
                        egui::Color32::from_rgb(230, 130, 120)
                    };
                    ui.colored_label(colour, format!("{:+.2e}", drift));
                }
                _ => {
                    ui.weak("baseline pending");
                }
            }
            ui.end_row();
        });
        ui.label(
            egui::RichText::new(
                "Energy is conserved by the continuum equations, so the drift is a \
                 direct measure of discretisation error. It steps whenever a bubble is \
                 stamped in or the potential is edited; the baseline re-zeroes after \
                 the last bubble.",
            )
            .small()
            .weak(),
        );
        if ui.button("Re-zero energy baseline").clicked() {
            state.energy_baseline = Some(diag.mean_energy);
        }
    });
}

fn physics(ui: &mut egui::Ui, model: &mut Model) {
    egui::CollapsingHeader::new("Physics").default_open(true).show(ui, |ui| {
        ui.add(
            egui::Slider::new(&mut model.eta, 0.0..=3.0)
                .text("friction  eta")
                .logarithmic(false),
        )
        .on_hover_text(
            "Drag of the plasma on the wall. Large eta gives a slow deflagration with \
             a shock running ahead of the wall; eta -> 0 gives a runaway detonation.",
        );

        ui.add(egui::Slider::new(&mut model.alpha, 0.005..=1.0).text("strength  alpha").logarithmic(true))
            .on_hover_text("Latent heat divided by the radiation energy density of the plasma.");

        ui.add(egui::Slider::new(&mut model.lambda, 0.05..=2.0).text("coupling  lambda").logarithmic(true))
            .on_hover_text("Sets the wall thickness and the surface tension.");

        ui.add(egui::Slider::new(&mut model.eps_ratio, 0.02..=0.95).text("barrier  eps ratio"))
            .on_hover_text(
                "Vacuum energy split as a fraction of its maximum. Small values mean a \
                 tall barrier and a large critical bubble.",
            );

        ui.add(egui::Slider::new(&mut model.cfl, 0.02..=0.4).text("Courant number"))
            .on_hover_text(
                "dt = cfl * dx. Stable up to 0.5, the limit the scalar field wave \
                 equation imposes on SSP-RK3, but accuracy degrades long before that: \
                 at 0.5 the energy drift is over an order of magnitude worse than at 0.2.",
            );

        ui.label(
            egui::RichText::new("Editing these changes the energy of the current state.")
                .small()
                .weak(),
        );

        ui.add_space(4.0);
        egui::Grid::new("derived").num_columns(2).striped(true).show(ui, |ui| {
            ui.label("wall thickness  l_w");
            ui.label(format!("{:.2} cells", model.wall_width()));
            ui.end_row();
            ui.label("critical radius  R_c");
            ui.label(format!("{:.1} cells", model.critical_radius()));
            ui.end_row();
            ui.label("surface tension  sigma");
            ui.label(format!("{:.4}", model.surface_tension()));
            ui.end_row();
            ui.label("bag constant  eps");
            ui.label(format!("{:.4}", model.eps()));
            ui.end_row();
            ui.label("sound speed  c_s");
            ui.label(format!("{:.3} c", model.sound_speed()));
            ui.end_row();
            ui.label("est. wall speed  v_w");
            ui.label(format!("{:.2} c", model.estimated_wall_velocity()));
            ui.end_row();
        });
        if model.wall_width() < 3.0 {
            ui.colored_label(
                egui::Color32::from_rgb(230, 160, 110),
                "Wall is thinner than 3 cells: expect lattice artefacts.",
            );
        }
    });
}

fn nucleation(ui: &mut egui::Ui, state: &mut UiState, model: &mut Model, cmd: &mut UiCommands) {
    egui::CollapsingHeader::new("Nucleation").default_open(true).show(ui, |ui| {
        egui::ComboBox::from_label("mode")
            .selected_text(match state.nucleation_mode {
                NucleationMode::Simultaneous => "simultaneous",
                NucleationMode::Exponential => "exponential",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut state.nucleation_mode,
                    NucleationMode::Simultaneous,
                    "simultaneous",
                );
                ui.selectable_value(
                    &mut state.nucleation_mode,
                    NucleationMode::Exponential,
                    "exponential",
                );
            });
        ui.add(egui::Slider::new(&mut state.bubble_count, 1..=200).text("bubbles"));
        ui.add_enabled(
            state.nucleation_mode == NucleationMode::Exponential,
            egui::Slider::new(&mut state.nucleation_duration, 5.0..=400.0).text("duration"),
        );
        ui.separator();
        seed_size(ui, model);

        ui.horizontal(|ui| {
            if ui.button("Apply (resets)").clicked() {
                cmd.reset = true;
            }
            if ui.button("Nucleate one now").clicked() {
                cmd.nucleate_now = true;
            }
        });
        ui.label(
            egui::RichText::new(
                "\"Nucleate one now\" uses the current seed size immediately, so you can \
                 watch a sub-critical bubble collapse without resetting.",
            )
            .small()
            .weak(),
        );
    });
}

/// Lattice size, with a memory budget that stops the user picking a size the
/// device cannot allocate.
///
/// Two independent ceilings apply. `max_storage_buffer_binding_size` is a hard
/// API cap on the `fluid` buffer at 16 bytes per cell, and no amount of free
/// memory relaxes it. The memory budget caps the total at 96 bytes per cell,
/// and is the one that usually binds on a desktop with other things running.
///
/// Overshooting either is not a recoverable error -- it is a lost device -- so
/// the slider is clamped to what fits rather than merely warning about it.
fn lattice(ui: &mut egui::Ui, state: &mut UiState, info: &SimInfo, cmd: &mut UiCommands) {
    egui::CollapsingHeader::new("Lattice").default_open(true).show(ui, |ui| {
        // Measured for what is running, predicted for what is selected.
        let current_bytes = info.device_bytes;
        let available = info.vram.map(|v| v.available());
        // The running lattice is released before the new one is allocated, so
        // its memory counts towards what a replacement may use.
        let budget = available.map(|a| vram::lattice_budget(a, current_bytes));

        let cap = sim::Simulation::max_cubic_size(info.max_storage_binding, budget);
        let max_axis = cap.side;

        ui.checkbox(&mut state.lattice_cubic, "cubic");
        if state.lattice_cubic {
            let mut n = state.lattice[0].clamp(sim::MIN_GRID, max_axis);
            ui.add(egui::Slider::new(&mut n, sim::MIN_GRID..=max_axis).text("cells per side"));
            state.lattice = [n; 3];
        } else {
            for (axis, name) in ["nx", "ny", "nz"].iter().enumerate() {
                ui.add(
                    egui::Slider::new(&mut state.lattice[axis], sim::MIN_GRID..=max_axis)
                        .text(*name),
                );
            }
        }
        for n in &mut state.lattice {
            *n = (*n).clamp(sim::MIN_GRID, max_axis);
        }

        let wanted = state.lattice;
        let wanted_bytes = sim::Simulation::lattice_bytes(wanted);
        let cells = wanted[0] as u64 * wanted[1] as u64 * wanted[2] as u64;

        egui::Grid::new("lattice").num_columns(2).striped(true).show(ui, |ui| {
            ui.label("selected");
            ui.label(format!(
                "{}x{}x{}  ({:.2} M cells)",
                wanted[0], wanted[1], wanted[2],
                cells as f64 / 1e6
            ));
            ui.end_row();

            ui.label("lattice needs");
            ui.label(format!("{:.2} GB", wanted_bytes as f64 / 1e9));
            ui.end_row();

            ui.label("running now");
            ui.label(format!(
                "{}x{}x{}  ({:.2} GB)",
                info.grid[0], info.grid[1], info.grid[2],
                current_bytes as f64 / 1e9
            ));
            ui.end_row();

            match info.vram {
                Some(v) => {
                    ui.label("device memory");
                    ui.label(format!("{:.1} GB total", v.capacity as f64 / 1e9));
                    ui.end_row();

                    ui.label(if v.available_is_measured() { "free now" } else { "free (est.)" });
                    ui.label(format!("{:.2} GB", v.available() as f64 / 1e9));
                    ui.end_row();

                    ui.label("budget for lattice");
                    ui.label(format!("{:.2} GB", budget.unwrap_or(0) as f64 / 1e9));
                    ui.end_row();
                }
                None => {
                    ui.label("device memory");
                    ui.weak("not reported by this backend");
                    ui.end_row();
                }
            }

            ui.label("largest cubic");
            ui.label(format!("{max_axis}^3"));
            ui.end_row();
        });

        // The slider is clamped, so these are advisory rather than the last
        // line of defence -- but a user who reaches the cap should be told why.
        if !cap.limited_by_memory {
            ui.label(
                egui::RichText::new(format!(
                    "Capped at {max_axis}^3 by max_storage_buffer_binding_size \
                     ({:.2} GB); free memory is not the constraint here.",
                    info.max_storage_binding as f64 / 1e9
                ))
                .small()
                .weak(),
            );
        } else {
            ui.label(
                egui::RichText::new(format!(
                    "Capped at {max_axis}^3 by free device memory, keeping {:.0}% headroom \
                     for the renderer and driver.",
                    100.0 * (1.0 - vram::LATTICE_BUDGET_FRACTION)
                ))
                .small()
                .weak(),
            );
        }

        if let Some(v) = info.vram {
            if wanted_bytes > v.capacity {
                ui.colored_label(
                    egui::Color32::from_rgb(230, 130, 120),
                    "Exceeds total device memory.",
                );
            } else if budget.is_some_and(|b| wanted_bytes > (b * 4) / 5) {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 200, 110),
                    "Close to the memory budget; other applications may be squeezed.",
                );
            }
            if !v.available_is_measured() {
                ui.label(
                    egui::RichText::new(
                        "This driver does not report a memory budget, so free memory is \
                         assumed to be the full capacity. Treat the cap as optimistic.",
                    )
                    .small()
                    .weak(),
                );
            }
        }

        let changed = wanted != info.grid;
        ui.horizontal(|ui| {
            if ui
                .add_enabled(changed, egui::Button::new("Apply (rebuilds & resets)"))
                .clicked()
            {
                cmd.resize = Some(wanted);
            }
            if ui.add_enabled(changed, egui::Button::new("Revert")).clicked() {
                state.select_lattice(info.grid);
            }
        });
        ui.label(
            egui::RichText::new(
                "Resizing reallocates every buffer and restarts the run. The old lattice \
                 is released first, so shrinking never fails for want of memory.",
            )
            .small()
            .weak(),
        );
    });
}

/// Seed-size controls.
///
/// Bubbles are stamped in rather than tunnelling, so their birth radius is a
/// free parameter -- and a physically loaded one.  A real thermal fluctuation
/// nucleates at exactly the critical radius, so the honest choice is a factor
/// just above 1; anything larger is a numerical convenience, and this panel
/// says so rather than hiding it behind a default.
fn seed_size(ui: &mut egui::Ui, model: &mut Model) {
    let r_c = model.critical_radius();
    let mut by_factor = matches!(model.seed_size, SeedSize::Critical(_));

    ui.horizontal(|ui| {
        ui.label("seed size");
        if ui.selectable_label(by_factor, "x R_c").clicked() {
            by_factor = true;
        }
        if ui.selectable_label(!by_factor, "cells").clicked() {
            by_factor = false;
        }
    });

    // Switching mode carries the current radius across, so the bubble does not
    // jump size the moment the user changes how it is specified.
    match (by_factor, model.seed_size) {
        (true, SeedSize::Cells(r)) => {
            model.seed_size = SeedSize::Critical((r / r_c.max(1e-6)).clamp(0.2, 6.0));
        }
        (false, SeedSize::Critical(k)) => {
            model.seed_size = SeedSize::Cells(k * r_c);
        }
        _ => {}
    }

    match &mut model.seed_size {
        SeedSize::Critical(k) => {
            ui.add(
                egui::Slider::new(k, 0.2..=6.0)
                    .text("R_0 / R_c")
                    .logarithmic(true),
            )
            .on_hover_text(
                "1.0 is the critical bubble: unstable equilibrium, so rounding error \
                 decides whether it grows. Below 1 it collapses. Just above 1 is the \
                 physically faithful choice.",
            );
        }
        SeedSize::Cells(r) => {
            ui.add(egui::Slider::new(r, 0.5..=60.0).text("R_0 (cells)").logarithmic(true));
        }
    }

    let r0 = model.seed_radius();
    egui::Grid::new("seed").num_columns(2).striped(true).show(ui, |ui| {
        ui.label("seed radius  R_0");
        ui.label(format!("{:.1} cells  ({:.2} R_c)", r0, r0 / r_c.max(1e-6)));
        ui.end_row();
        ui.label("R_c / l_w");
        ui.label(format!("{:.1}", model.critical_to_wall_ratio()));
        ui.end_row();
    });

    // Two independent failure modes, worth calling out separately.
    if r0 < r_c {
        ui.colored_label(
            egui::Color32::from_rgb(150, 190, 240),
            "Sub-critical: these bubbles will shrink and vanish.",
        );
    } else if r0 < 1.05 * r_c {
        ui.colored_label(
            egui::Color32::from_rgb(220, 200, 110),
            "Within 5% of critical: growth or collapse is decided by rounding.",
        );
    }
    if r0 < model.wall_width() {
        ui.colored_label(
            egui::Color32::from_rgb(230, 160, 110),
            "Seed is thinner than the wall; the profile is not resolved.",
        );
    }
    if model.critical_to_wall_ratio() < 3.0 {
        ui.label(
            egui::RichText::new(
                "R_c is only a few wall widths, so this is a thick-wall bubble and \
                 R_c = 2 sigma / eps is indicative rather than exact. Lower eps ratio \
                 for a cleaner thin-wall regime.",
            )
            .small()
            .weak(),
        );
    }
}

fn visualisation(ui: &mut egui::Ui, render: &mut RenderSettings, cmd: &mut UiCommands) {
    ui.collapsing("Visualisation", |ui| {
        let before = render.field_mode;
        egui::ComboBox::from_label("volume field")
            .selected_text(render.field_mode.label())
            .show_ui(ui, |ui| {
                for m in FieldMode::ALL {
                    ui.selectable_value(&mut render.field_mode, m, m.label());
                }
            });
        if render.field_mode != before {
            render.field_gain = render.field_mode.default_gain();
            render.colormap = render.field_mode.default_colormap();
        }

        egui::ComboBox::from_label("colour map")
            .selected_text(COLORMAPS[render.colormap.min(COLORMAPS.len() - 1)])
            .show_ui(ui, |ui| {
                for (i, name) in COLORMAPS.iter().enumerate() {
                    ui.selectable_value(&mut render.colormap, i, *name);
                }
            });

        ui.checkbox(&mut render.show_volume, "show plasma");
        ui.add_enabled(
            render.show_volume,
            egui::Slider::new(&mut render.field_gain, 0.1..=200.0).text("gain").logarithmic(true),
        );
        ui.add_enabled(
            render.show_volume,
            egui::Slider::new(&mut render.absorption, 1.0..=500.0)
                .text("opacity")
                .logarithmic(true),
        );

        ui.checkbox(&mut render.show_iso, "show bubble walls");
        ui.add_enabled(
            render.show_iso,
            egui::Slider::new(&mut render.iso_level, 0.05..=0.95).text("wall level  phi/phi_b"),
        );
        ui.add_enabled(
            render.show_iso,
            egui::Slider::new(&mut render.wall_opacity, 0.0..=1.0).text("wall opacity"),
        );

        ui.add(egui::Slider::new(&mut render.exposure, 0.05..=5.0).text("exposure").logarithmic(true));
        ui.add(egui::Slider::new(&mut render.samples_per_cell, 0.25..=4.0).text("samples / cell"))
            .on_hover_text("Ray-march density. Lower is faster and grainier.");

        ui.checkbox(&mut render.show_box, "show box outline");

        ui.horizontal(|ui| {
            ui.label("cutaway");
            for (i, name) in ["off", "x", "y", "z"].iter().enumerate() {
                let axis = i as i32 - 1;
                if ui.selectable_label(render.clip_axis == axis, *name).clicked() {
                    render.clip_axis = axis;
                }
            }
        });
        ui.add_enabled(
            render.clip_axis >= 0,
            egui::Slider::new(&mut render.clip_pos, -1.0..=1.0).text("cut position"),
        );

        if ui.button("Reset view").clicked() {
            cmd.frame_view = true;
        }
    });
}

fn help(ui: &mut egui::Ui) {
    ui.collapsing("Controls & model", |ui| {
        ui.label(
            egui::RichText::new(
                "Mouse\n\
                 \u{2022} drag left: orbit\n\
                 \u{2022} drag right/middle: pan\n\
                 \u{2022} wheel: zoom\n\n\
                 Keys\n\
                 \u{2022} space: run/pause    \u{2022} . : single step\n\
                 \u{2022} W/S: fly forward/back    \u{2022} R: reset simulation\n\
                 \u{2022} N: nucleate a bubble    \u{2022} F: reset view\n\
                 \u{2022} H: hide this panel     \u{2022} Esc: quit",
            )
            .small(),
        );
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new(
                "A scalar order parameter phi coupled to a relativistic plasma with the \
                 bag equation of state (e = 3p, c_s = 1/sqrt(3)). Bubbles of the broken \
                 phase are seeded above the critical radius; the vacuum energy eps drives \
                 their walls outward, friction eta and the fluid back-reaction set the \
                 terminal wall speed, and the walls stir the plasma. What is left after \
                 the bubbles merge is a field of sound waves - the dominant source of \
                 gravitational waves from a cosmological phase transition.",
            )
            .small()
            .weak(),
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sim_info() -> SimInfo {
        SimInfo {
            grid: [192; 3],
            time: 12.5,
            steps: 100,
            cells: 192 * 192 * 192,
            bubbles_remaining: 3,
            device_bytes: 680_000_000,
            vram: Some(VramInfo {
                capacity: 25_769_803_776,
                budget: Some(24_000_000_000),
                used: Some(2_000_000_000),
            }),
            max_storage_binding: 2_147_483_648,
        }
    }

    /// Every string the panel actually draws, pulled back out of the laid-out
    /// shapes. Lets a test assert on what the user sees rather than on the code
    /// that produces it.
    fn rendered_text(state: &mut UiState) -> String {
        fn collect(shape: &egui::epaint::Shape, out: &mut String) {
            match shape {
                egui::epaint::Shape::Text(t) => {
                    out.push_str(t.galley.text());
                    out.push('\n');
                }
                egui::epaint::Shape::Vec(v) => v.iter().for_each(|s| collect(s, out)),
                _ => {}
            }
        }

        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::pos2(0.0, 0.0),
                egui::vec2(1600.0, 900.0),
            )),
            ..Default::default()
        };
        let mut model = Model::default();
        let mut render = RenderSettings::default();
        let info = sim_info();
        let diag = Diagnostics::default();
        let mut cmd = UiCommands::default();

        let output = ctx.run_ui(input, |ui| {
            draw(ui, state, &mut model, &mut render, &info, &diag, &mut cmd);
        });

        let mut text = String::new();
        for clipped in &output.shapes {
            collect(&clipped.shape, &mut text);
        }
        text
    }

    /// The frame time must be on screen, in milliseconds, without the user
    /// having to expand anything.
    #[test]
    fn frame_time_in_milliseconds_is_always_visible() {
        let mut state = UiState { frame_ms: 7.25, frame_ms_max: 9.5, fps: 138.0, ..Default::default() };
        let text = rendered_text(&mut state);
        assert!(text.contains("7.2"), "frame time missing from the panel:\n{text}");
        assert!(text.contains("ms / frame"), "no ms/frame label:\n{text}");
        assert!(text.contains("138 fps"), "fps missing:\n{text}");
        assert!(text.contains("worst 9.5 ms"), "worst-frame figure missing:\n{text}");
    }

    /// It is a top-level readout, not something hidden inside a collapsed
    /// section -- which is the whole point of moving it.
    #[test]
    fn frame_time_survives_every_section_being_collapsed() {
        let mut state = UiState { frame_ms: 4.0, ..Default::default() };
        // A fresh egui context starts with collapsing headers in their default
        // state; the timing line sits above all of them regardless.
        let text = rendered_text(&mut state);
        assert!(text.contains("ms / frame"));
    }

    /// Dividing by `steps_per_frame` must not blow up if it is somehow zero.
    #[test]
    fn per_step_timing_tolerates_a_zero_step_count() {
        let mut state =
            UiState { frame_ms: 8.0, steps_per_frame: 0, running: true, ..Default::default() };
        let text = rendered_text(&mut state);
        assert!(text.contains("8.00 ms / step"), "expected 8 ms/step fallback:\n{text}");
    }

    /// While paused the frame time is pure rendering, so a "ms / step" figure
    /// would be a lie.
    #[test]
    fn per_step_timing_is_withheld_while_paused() {
        let mut state = UiState { frame_ms: 8.0, running: false, ..Default::default() };
        let text = rendered_text(&mut state);
        assert!(text.contains("ms / frame"), "frame time should still show:\n{text}");
        assert!(!text.contains("ms / step"), "no per-step figure while paused:\n{text}");
        assert!(text.contains("paused"));
    }

    /// A non-cubic grid must not leave the cubic editor enabled, or the panel
    /// rewrites the selection to `[n; 3]` on the next frame and shows a pending
    /// change the user never asked for.
    #[test]
    fn selecting_a_non_cubic_grid_clears_the_cubic_toggle() {
        let mut state = UiState::default();
        state.select_lattice([256, 128, 128]);
        assert_eq!(state.lattice, [256, 128, 128]);
        assert!(!state.lattice_cubic);

        state.select_lattice([192; 3]);
        assert!(state.lattice_cubic);
    }
}
