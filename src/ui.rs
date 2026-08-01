//! Live control panel.
//!
//! Everything here is either a solver parameter that is safe to change mid-run
//! (the shaders read them from a uniform, so the next dispatch simply uses new
//! constants) or a rendering choice.  Parameters that change the potential also
//! change the energy of the current configuration, which is called out in the
//! panel rather than hidden.

use crate::gpu::render::{COLORMAPS, FieldMode, RenderSettings};
use crate::gpu::sim::Diagnostics;
use crate::physics::{Model, NucleationMode};

pub struct UiState {
    pub visible: bool,
    pub running: bool,
    pub steps_per_frame: u32,
    pub nucleation_mode: NucleationMode,
    pub bubble_count: usize,
    pub nucleation_duration: f32,
    pub seed: u64,
    /// Total energy at the moment the last nucleation finished; the drift
    /// readout is measured against this.
    pub energy_baseline: Option<f32>,
    pub fps: f32,
    pub gpu_frame_ms: f32,
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
            energy_baseline: None,
            fps: 0.0,
            gpu_frame_ms: 0.0,
        }
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
}

pub struct SimInfo {
    pub grid: [u32; 3],
    pub time: f32,
    pub steps: u64,
    pub cells: u64,
    pub bubbles_remaining: usize,
    pub device_bytes: u64,
}

pub fn draw(
    root: &mut egui::Ui,
    state: &mut UiState,
    model: &mut Model,
    render: &mut RenderSettings,
    info: &SimInfo,
    diag: &Diagnostics,
) -> UiCommands {
    let mut cmd = UiCommands::default();
    if !state.visible {
        return cmd;
    }

    egui::Panel::left("controls")
        .resizable(true)
        .default_size(340.0)
        .show(root, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                transport(ui, state, &mut cmd);
                ui.separator();
                diagnostics(ui, state, info, diag, model);
                ui.separator();
                physics(ui, model);
                ui.separator();
                nucleation(ui, state, model, &mut cmd);
                ui.separator();
                visualisation(ui, render, &mut cmd);
                ui.separator();
                help(ui);
            });
        });

    cmd
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
    ui.label(
        egui::RichText::new(format!(
            "{:.0} fps   |   {:.1} ms / frame",
            state.fps, state.gpu_frame_ms
        ))
        .small()
        .weak(),
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

fn nucleation(ui: &mut egui::Ui, state: &mut UiState, model: &Model, cmd: &mut UiCommands) {
    ui.collapsing("Nucleation", |ui| {
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
        ui.label(
            egui::RichText::new(format!("seed radius {:.1} cells", model.default_seed_radius()))
                .small()
                .weak(),
        );
        ui.horizontal(|ui| {
            if ui.button("Apply (resets)").clicked() {
                cmd.reset = true;
            }
            if ui.button("Nucleate one now").clicked() {
                cmd.nucleate_now = true;
            }
        });
    });
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
