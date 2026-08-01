//! Window, device, and the per-frame loop.
//!
//! The frame is a single command submission:
//!
//!   1. `n` SSP-RK3 time steps (plus any nucleation that has come due),
//!   2. one pass packing the state into the visualisation 3D texture,
//!   3. optionally the global reduction behind the diagnostics readout,
//!   4. the volume ray-march into the swapchain image,
//!   5. the egui overlay.
//!
//! Steps 1-4 read and write device memory only.  Nothing about the lattice
//! travels over PCIe in either direction; the sole exception is a 32-byte
//! diagnostics struct fetched asynchronously for the numbers in the panel.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use glam::Vec2;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use crate::camera::OrbitCamera;
use crate::config::Config;
use crate::gpu::render::{RenderSettings, VolumeRenderer};
use crate::gpu::sim::{Diagnostics, Simulation};
use crate::physics::{Model, make_schedule};
use crate::ui::{self, SimInfo, UiCommands, UiState};

#[derive(Default)]
struct Input {
    orbiting: bool,
    panning: bool,
    last_cursor: Option<Vec2>,
    fly_forward: f32,
}

pub struct App {
    config: Config,
    state: Option<State>,
}

impl App {
    pub fn new(config: Config) -> Self {
        Self { config, state: None }
    }
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,

    sim: Simulation,
    renderer: VolumeRenderer,
    camera: OrbitCamera,

    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,

    ui: UiState,
    render_settings: RenderSettings,
    model: Model,
    diagnostics: Diagnostics,

    input: Input,
    rng: Pcg64Mcg,
    last_frame: Instant,
    frame_times: VecDeque<f32>,
    /// Requested by a key press; consumed by the next redraw.
    queued: UiCommands,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        match pollster::block_on(State::new(event_loop, &self.config)) {
            Ok(state) => self.state = Some(state),
            Err(err) => {
                log::error!("{err:?}");
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else { return };
        state.window_event(event_loop, event);
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }
}

impl State {
    async fn new(event_loop: &ActiveEventLoop, config: &Config) -> Result<Self> {
        let attrs = Window::default_attributes()
            .with_title("Bubble nucleation in a first-order cosmological phase transition")
            .with_inner_size(winit::dpi::LogicalSize::new(1600.0, 950.0));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .context("failed to create the window")?,
        );

        // The `display` handle is only consulted by the GL backend; Vulkan,
        // which is what this targets, does not need it.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance
            .create_surface(window.clone())
            .context("failed to create a surface for the window")?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("no suitable GPU adapter found")?;
        let info = adapter.get_info();
        log::info!("adapter: {} ({:?}, {:?})", info.name, info.backend, info.device_type);

        // The lattice buffers are far larger than the WebGPU default limits
        // allow (which cap a storage binding at 128 MiB), so take what this
        // adapter actually offers.  On an RTX 3090 that is a 24 GB budget.
        let limits = adapter.limits();
        log::info!(
            "limits: {:.0} MB per storage binding, {:.1} GB per buffer",
            limits.max_storage_buffer_binding_size as f64 / 1e6,
            limits.max_buffer_size as f64 / 1e9,
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("bubble simulation"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .context("failed to acquire a device with the requested limits")?;

        device.on_uncaptured_error(Arc::new(|err| log::error!("wgpu: {err}")));

        // Deliberately pick a non-sRGB swapchain format: the volume shader
        // emits gamma-encoded colour and egui has a matching path for that.
        let caps = surface.get_capabilities(&adapter);
        anyhow::ensure!(!caps.formats.is_empty(), "surface reports no supported formats");
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| *f == wgpu::TextureFormat::Bgra8Unorm)
            .or_else(|| caps.formats.iter().copied().find(|f| !f.is_srgb()))
            .unwrap_or(caps.formats[0]);

        let present_mode = if config.no_vsync {
            wgpu::PresentMode::AutoNoVsync
        } else {
            wgpu::PresentMode::AutoVsync
        };
        let size = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        let model = config.model();
        let grid = config.grid();
        let sim = Simulation::new(&device, &queue, config.sim_spec())
            .context("failed to create the simulation")?;

        let renderer = VolumeRenderer::new(&device, format, &sim.vis_view, grid);
        let mut camera = OrbitCamera::default();
        camera.frame_box(renderer.box_half());

        let egui_ctx = egui::Context::default();
        egui_ctx.set_visuals(egui::Visuals::dark());
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            None,
            None,
            None,
        );
        let egui_renderer =
            egui_wgpu::Renderer::new(&device, format, egui_wgpu::RendererOptions::default());

        let ui = UiState {
            nucleation_mode: config.nucleation,
            bubble_count: config.bubbles,
            nucleation_duration: config.nucleation_duration,
            seed: config.seed,
            steps_per_frame: config.steps_per_frame,
            ..UiState::default()
        };

        log::info!(
            "lattice {}x{}x{} = {:.2} M cells | dt = {:.3} | wall {:.1} cells | R_c {:.1} cells",
            grid[0],
            grid[1],
            grid[2],
            sim.cell_count() as f64 / 1e6,
            model.dt(),
            model.wall_width(),
            model.critical_radius(),
        );

        Ok(Self {
            window,
            surface,
            device,
            queue,
            surface_config,
            sim,
            renderer,
            camera,
            egui_ctx,
            egui_state,
            egui_renderer,
            ui,
            render_settings: RenderSettings::with_field(config.field),
            model,
            diagnostics: Diagnostics::default(),
            input: Input::default(),
            rng: Pcg64Mcg::seed_from_u64(config.seed ^ 0x9E37_79B9_7F4A_7C15),
            last_frame: Instant::now(),
            frame_times: VecDeque::with_capacity(64),
            queued: UiCommands::default(),
        })
    }

    // -----------------------------------------------------------------------
    //  Input
    // -----------------------------------------------------------------------

    fn window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        let response = self.egui_state.on_window_event(&self.window, &event);
        let ui_wants_input = response.consumed;

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                self.surface_config.width = size.width.max(1);
                self.surface_config.height = size.height.max(1);
                self.surface.configure(&self.device, &self.surface_config);
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if !ui_wants_input {
                    self.on_key(event_loop, &event);
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                if pressed && ui_wants_input {
                    return;
                }
                match button {
                    MouseButton::Left => self.input.orbiting = pressed,
                    MouseButton::Right | MouseButton::Middle => self.input.panning = pressed,
                    _ => {}
                }
                if !pressed {
                    self.input.last_cursor = None;
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let now = Vec2::new(position.x as f32, position.y as f32);
                if let Some(prev) = self.input.last_cursor {
                    let d = now - prev;
                    if self.input.orbiting {
                        self.camera.orbit(d.x * 0.006, d.y * 0.006);
                    } else if self.input.panning {
                        let h = self.surface_config.height.max(1) as f32;
                        self.camera.pan(d.x / h, d.y / h);
                    }
                }
                self.input.last_cursor =
                    (self.input.orbiting || self.input.panning).then_some(now);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if ui_wants_input {
                    return;
                }
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                self.camera.zoom(amount);
            }

            WindowEvent::RedrawRequested => self.redraw(),

            _ => {}
        }
    }

    fn on_key(&mut self, event_loop: &ActiveEventLoop, event: &winit::event::KeyEvent) {
        let pressed = event.state == ElementState::Pressed;
        let fresh = pressed && !event.repeat;
        match &event.logical_key {
            Key::Named(NamedKey::Escape) if pressed => event_loop.exit(),
            Key::Named(NamedKey::Space) if fresh => self.ui.running = !self.ui.running,
            Key::Character(c) => {
                let held = if pressed { 1.0 } else { 0.0 };
                match c.as_str() {
                    "w" | "W" => self.input.fly_forward = held,
                    "s" | "S" => self.input.fly_forward = -held,
                    "." if pressed => self.queued.single_step = true,
                    "r" | "R" if fresh => self.queued.reset = true,
                    "n" | "N" if fresh => self.queued.nucleate_now = true,
                    "f" | "F" if fresh => self.queued.frame_view = true,
                    "h" | "H" if fresh => self.ui.visible = !self.ui.visible,
                    _ => {}
                }
            }
            _ => {}
        }
    }

    // -----------------------------------------------------------------------
    //  Frame
    // -----------------------------------------------------------------------

    fn reset(&mut self, new_seed: bool) {
        if new_seed {
            self.ui.seed = self
                .ui
                .seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
        }
        let schedule = make_schedule(
            self.ui.nucleation_mode,
            self.ui.bubble_count,
            self.model.default_seed_radius(),
            self.sim.grid,
            self.ui.nucleation_duration,
            self.ui.seed,
        );
        self.sim.set_model(&self.queue, self.model);
        self.sim.reset(&self.device, &self.queue, Some(schedule));
        self.ui.energy_baseline = None;
    }

    fn redraw(&mut self) {
        // ---- timing ---------------------------------------------------------
        let now = Instant::now();
        let dt_wall = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        if self.frame_times.len() == 64 {
            self.frame_times.pop_front();
        }
        self.frame_times.push_back(dt_wall);
        let mean = self.frame_times.iter().sum::<f32>() / self.frame_times.len().max(1) as f32;
        self.ui.fps = if mean > 0.0 { 1.0 / mean } else { 0.0 };
        self.ui.gpu_frame_ms = mean * 1000.0;

        if self.input.fly_forward != 0.0 {
            self.camera.dolly(self.input.fly_forward * dt_wall * 1.2);
        }

        // ---- UI -------------------------------------------------------------
        self.diagnostics = self.sim.poll_diagnostics();
        let mut model_edit = self.model;
        let info = SimInfo {
            grid: self.sim.grid,
            time: self.sim.time,
            steps: self.sim.steps,
            cells: self.sim.cell_count(),
            bubbles_remaining: self.sim.remaining_bubbles(),
            device_bytes: self.sim.device_memory_bytes(),
        };

        let ctx = self.egui_ctx.clone();
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let mut cmd = std::mem::take(&mut self.queued);
        let ui_state = &mut self.ui;
        let render_settings = &mut self.render_settings;
        let diagnostics = &self.diagnostics;
        let full_output = ctx.run_ui(raw_input, |root| {
            let from_ui = ui::draw(
                root,
                ui_state,
                &mut model_edit,
                render_settings,
                &info,
                diagnostics,
            );
            cmd.reset |= from_ui.reset;
            cmd.reseed_and_reset |= from_ui.reseed_and_reset;
            cmd.nucleate_now |= from_ui.nucleate_now;
            cmd.single_step |= from_ui.single_step;
            cmd.frame_view |= from_ui.frame_view;
        });
        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);

        if model_edit != self.model {
            self.model = model_edit;
            self.sim.set_model(&self.queue, self.model);
            self.ui.energy_baseline = None;
        }

        // ---- act on commands -------------------------------------------------
        if cmd.reset || cmd.reseed_and_reset {
            self.reset(cmd.reseed_and_reset);
        }
        if cmd.nucleate_now {
            self.sim.nucleate_now(&mut self.rng);
        }
        if cmd.frame_view {
            self.camera.frame_box(self.renderer.box_half());
        }
        if cmd.single_step {
            self.ui.running = false;
        }

        // ---- acquire the swapchain image -------------------------------------
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            other => {
                log::debug!("skipping frame: {other:?}");
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let aspect = self.surface_config.width as f32 / self.surface_config.height.max(1) as f32;
        self.renderer.update(
            &self.queue,
            self.camera.view_proj(aspect),
            self.camera.eye(),
            &self.render_settings,
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        // ---- simulate ---------------------------------------------------------
        let steps = if self.ui.running {
            self.ui.steps_per_frame
        } else if cmd.single_step {
            1
        } else {
            0
        };
        if steps > 0 {
            self.sim.record_steps(&self.queue, &mut encoder, steps);
        }
        let nucleated = self.sim.nucleated_this_frame;
        self.sim.record_visualisation(&mut encoder);
        self.sim.record_diagnostics(&mut encoder);

        // ---- render ------------------------------------------------------------
        self.renderer.draw(&mut encoder, &view);

        let pixels_per_point = self.egui_ctx.pixels_per_point();
        let paint_jobs = self.egui_ctx.tessellate(full_output.shapes, pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point,
        };
        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        let user_cmds = self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &screen,
        );
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            self.egui_renderer.render(&mut pass, &paint_jobs, &screen);
        }

        self.queue
            .submit(user_cmds.into_iter().chain(std::iter::once(encoder.finish())));
        frame.present();

        // Only now may the staging buffer be mapped.
        self.sim.after_submit();
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        // Non-blocking poll so the readback callback can fire; the numbers
        // arrive a frame or two late, which is invisible in the UI.
        let _ = self.device.poll(wgpu::PollType::Poll);

        // Stamping a bubble injects its surface and vacuum energy by hand, so
        // the conservation baseline is only meaningful once nucleation is over.
        if nucleated {
            self.ui.energy_baseline = None;
        } else if self.ui.energy_baseline.is_none()
            && self.sim.remaining_bubbles() == 0
            && self.diagnostics.mean_energy != 0.0
        {
            self.ui.energy_baseline = Some(self.diagnostics.mean_energy);
        }
    }
}
