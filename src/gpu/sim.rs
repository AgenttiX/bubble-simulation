//! The simulation itself: GPU buffers, compute pipelines, and the time step.
//!
//! All state lives in device memory for the whole run.  The only host readback
//! is a 32-byte diagnostics struct, fetched asynchronously and one frame late,
//! purely so the UI can print numbers; see [`Simulation::poll_diagnostics`].
//!
//! # Memory layout
//!
//! Six evolved fields per cell, split into two buffers so that both are
//! naturally aligned and coalesce well:
//!
//! | buffer  | contents            | bytes/cell |
//! |---------|---------------------|------------|
//! | `field` | `(phi, pi)`         | 8          |
//! | `fluid` | `(E, Zx, Zy, Zz)`   | 16         |
//! | `prim`  | `(vx, vy, vz, p)`   | 16 (cache) |
//!
//! Three copies of `(field, fluid)` are kept, which is the minimum for
//! SSP-RK3: the stage combination `a0 U^n + a1 (U^k + dt L(U^k))` reads `U^n`
//! and `U^k` while writing a third slot.  At 192^3 that is 510 MB; at 256^3,
//! 1.2 GB - both comfortable on a 24 GB RTX 3090.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::Result;
use bytemuck::{Pod, Zeroable};

use crate::gpu::preprocess::{self, Precision};
use crate::physics::{BubbleEvent, Model, NucleationMode, Scalar, make_schedule};

/// Threads per workgroup along x for the lattice passes.  Consecutive threads
/// then touch consecutive linear indices, which is what the memory system wants.
const WG_X: u32 = 64;
/// Number of partial sums produced by the first reduction pass.
/// Must match `N_PARTIALS` in `shaders/reduce.wgsl`.
const N_PARTIALS: u32 = 1024;
/// Workgroup size of the reduction passes.  Must match `WG_SIZE` there.
const REDUCE_WG: u32 = 256;
/// Upper bound on bubbles stamped in a single nucleation dispatch.
const MAX_BUBBLES: usize = 1024;

const DIAG_BYTES: u64 = 32;

/// Smallest lattice the MUSCL stencil can address without a cell reaching its
/// own periodic image.
pub const MIN_GRID: u32 = 8;

/// Upper bound offered in the UI. Well past what any current device can hold;
/// it exists so the slider range stays finite when memory cannot be queried.
pub const MAX_GRID: u32 = 2048;

/// The largest cubic lattice that fits, and why it is the largest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatticeCap {
    pub side: u32,
    /// True when free memory is the binding constraint rather than the
    /// device's storage-buffer binding limit.
    pub limited_by_memory: bool,
}

// ---------------------------------------------------------------------------
//  GPU-side parameter blocks (must mirror shaders/common.wgsl exactly)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SimParamsGpu {
    nx: u32,
    ny: u32,
    nz: u32,
    _pad_u0: u32,

    dx: f32,
    inv_dx: f32,
    dt: f32,
    eta: f32,

    lambda: f32,
    eps: f32,
    phi_b: f32,
    m2: f32,

    delta: f32,
    a_rad: f32,
    wall_width: f32,
    p_floor: f32,

    e_floor: f32,
    z_max: f32,
    e_ref: f32,
    t_ref: f32,

    time: f32,
    vis_gain: f32,
    _pad0: f32,
    _pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct StageParamsGpu {
    a0: f32,
    a1: f32,
    dt_stage: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct BubbleGpu {
    cx: f32,
    cy: f32,
    cz: f32,
    r0: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct NucleationBatchGpu {
    count: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct DiagnosticsGpu {
    sums: [f32; 4],
    max_v_bits: u32,
    _pad: [u32; 3],
}

/// Strong-stability-preserving RK3 (Shu-Osher).  Each stage is
/// `U_out = a0 U^n + a1 (U_in + dt L(U_in))`.
const RK3: [(f32, f32); 3] = [(0.0, 1.0), (0.75, 0.25), (1.0 / 3.0, 2.0 / 3.0)];

// ---------------------------------------------------------------------------
//  Host-visible diagnostics
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
pub struct Diagnostics {
    /// Volume-averaged total energy density (field + fluid).  Conserved by the
    /// continuum equations, so its drift measures discretisation error.
    pub mean_energy: f32,
    /// Fraction of the box in the broken phase.
    pub broken_fraction: f32,
    /// Volume-averaged fluid kinetic energy density `w W^2 v^2`.
    pub mean_kinetic: f32,
    /// Volume-averaged fluid energy density `E`.
    pub mean_fluid_energy: f32,
    /// Largest fluid three-velocity anywhere in the box.
    pub max_velocity: f32,
}

impl Diagnostics {
    /// Kinetic energy fraction `K`: the share of the total energy carried by
    /// bulk fluid motion.  This is the quantity that controls gravitational
    /// wave production from a phase transition.
    pub fn kinetic_fraction(&self) -> f32 {
        if self.mean_energy.abs() < 1e-20 {
            0.0
        } else {
            self.mean_kinetic / self.mean_energy
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReadbackState {
    Idle = 0,
    Pending = 1,
    Ready = 2,
}

// ---------------------------------------------------------------------------
//  Simulation
// ---------------------------------------------------------------------------

pub struct Simulation {
    pub model: Model,
    pub grid: [u32; 3],
    /// The spec this was built from, so a resize can rebuild with one field
    /// changed rather than reassembling the arguments at the call site.
    spec: SimulationSpec,
    pub time: Scalar,
    pub steps: u64,
    /// Visualisation gain applied inside `vis.wgsl`; currently unused headroom
    /// for future channel scaling.
    pub vis_gain: f32,

    n_cells: u64,

    // --- schedule -----------------------------------------------------------
    schedule: Vec<BubbleEvent>,
    next_event: usize,
    /// Set on the frame a nucleation dispatch happens, so the UI can re-baseline
    /// the energy-conservation readout (stamping a bubble injects energy).
    pub nucleated_this_frame: bool,

    // --- buffers ------------------------------------------------------------
    field: [wgpu::Buffer; 3],
    fluid: [wgpu::Buffer; 3],
    prim: wgpu::Buffer,
    partials: wgpu::Buffer,
    diag: wgpu::Buffer,
    diag_staging: wgpu::Buffer,
    bubbles: wgpu::Buffer,
    batch_ub: wgpu::Buffer,
    params_ub: wgpu::Buffer,
    stage_ub: [wgpu::Buffer; 3],

    pub vis_view: wgpu::TextureView,
    _vis_tex: wgpu::Texture,

    // --- pipelines ----------------------------------------------------------
    pipe_init: wgpu::ComputePipeline,
    pipe_nucleate: wgpu::ComputePipeline,
    pipe_prim: wgpu::ComputePipeline,
    pipe_step: wgpu::ComputePipeline,
    pipe_vis: wgpu::ComputePipeline,
    pipe_reduce1: wgpu::ComputePipeline,
    pipe_reduce2: wgpu::ComputePipeline,

    // --- bind groups --------------------------------------------------------
    bg_params: wgpu::BindGroup,
    bg_params_stage: [wgpu::BindGroup; 3],
    bg_init: [wgpu::BindGroup; 3],
    bg_nucleate: [wgpu::BindGroup; 3],
    bg_prim: [wgpu::BindGroup; 3],
    /// `bg_step[current_slot][stage]`
    bg_step: [[wgpu::BindGroup; 3]; 3],
    bg_vis: [wgpu::BindGroup; 3],
    bg_reduce: [wgpu::BindGroup; 3],

    /// Which of the three slots currently holds `U^n`.
    cur: usize,

    readback_state: Arc<AtomicU32>,
    diagnostics: Diagnostics,
    copy_issued: bool,
}

/// Everything needed to construct a [`Simulation`], gathered so the call site
/// reads as named fields rather than nine positional arguments.
#[derive(Clone, Copy, Debug)]
pub struct SimulationSpec {
    pub model: Model,
    pub grid: [u32; 3],
    pub nucleation: NucleationMode,
    pub bubbles: usize,
    pub nucleation_duration: Scalar,
    pub seed: u64,
    pub precision: Precision,
}

impl Simulation {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, spec: SimulationSpec) -> Result<Self> {
        let SimulationSpec { model, grid, precision, .. } = spec;
        model.validate().map_err(anyhow::Error::msg)?;
        Self::check_fits(device, grid)?;

        let n_cells = grid[0] as u64 * grid[1] as u64 * grid[2] as u64;
        let field_bytes = n_cells * 8;
        let fluid_bytes = n_cells * 16;

        // ---- shader modules -------------------------------------------------
        let module = |name: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(preprocess::build(name, precision).into()),
            })
        };
        let m_init = module("init");
        let m_nucleate = module("nucleate");
        let m_prim = module("primitives");
        let m_step = module("step");
        let m_vis = module("vis");
        let m_reduce = module("reduce");

        // ---- buffers --------------------------------------------------------
        let storage = wgpu::BufferUsages::STORAGE;
        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };

        let field = [
            mk("field 0", field_bytes, storage),
            mk("field 1", field_bytes, storage),
            mk("field 2", field_bytes, storage),
        ];
        let fluid = [
            mk("fluid 0", fluid_bytes, storage),
            mk("fluid 1", fluid_bytes, storage),
            mk("fluid 2", fluid_bytes, storage),
        ];
        let prim = mk("primitives", fluid_bytes, storage);
        let partials = mk("reduction partials", N_PARTIALS as u64 * 16, storage);
        let diag = mk(
            "diagnostics",
            DIAG_BYTES,
            storage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let diag_staging = mk(
            "diagnostics staging",
            DIAG_BYTES,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let bubbles = mk(
            "bubbles",
            (MAX_BUBBLES * std::mem::size_of::<BubbleGpu>()) as u64,
            storage | wgpu::BufferUsages::COPY_DST,
        );

        let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let params_ub = mk("sim params", std::mem::size_of::<SimParamsGpu>() as u64, uniform);
        let batch_ub = mk("nucleation batch", 16, uniform);
        let stage_sz = std::mem::size_of::<StageParamsGpu>() as u64;
        let stage_ub = [
            mk("rk stage 0", stage_sz, uniform),
            mk("rk stage 1", stage_sz, uniform),
            mk("rk stage 2", stage_sz, uniform),
        ];

        // ---- visualisation texture ------------------------------------------
        let vis_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("visualisation volume"),
            size: wgpu::Extent3d {
                width: grid[0],
                height: grid[1],
                depth_or_array_layers: grid[2],
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let vis_view = vis_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // ---- bind group layouts ---------------------------------------------
        let ub_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let sb_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bgl_params = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("params"),
            entries: &[ub_entry(0)],
        });
        let bgl_params_stage = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("params + stage"),
            entries: &[ub_entry(0), ub_entry(1)],
        });
        let bgl_init = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("init"),
            entries: &[sb_entry(0, false), sb_entry(1, false)],
        });
        let bgl_nucleate = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nucleate"),
            entries: &[sb_entry(0, false), sb_entry(1, true), ub_entry(2)],
        });
        let bgl_prim = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("primitives"),
            entries: &[sb_entry(0, true), sb_entry(1, true), sb_entry(2, false)],
        });
        let bgl_step = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("step"),
            entries: &[
                sb_entry(0, true),
                sb_entry(1, true),
                sb_entry(2, true),
                sb_entry(3, true),
                sb_entry(4, true),
                sb_entry(5, false),
                sb_entry(6, false),
            ],
        });
        let bgl_vis = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vis"),
            entries: &[
                sb_entry(0, true),
                sb_entry(1, true),
                sb_entry(2, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });
        let bgl_reduce = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reduce"),
            entries: &[
                sb_entry(0, true),
                sb_entry(1, true),
                sb_entry(2, true),
                sb_entry(3, false),
                sb_entry(4, false),
            ],
        });

        // ---- pipelines -------------------------------------------------------
        let layout = |label: &str, groups: &[Option<&wgpu::BindGroupLayout>]| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: groups,
                immediate_size: 0,
            })
        };
        let compute = |label: &str,
                       layout: &wgpu::PipelineLayout,
                       module: &wgpu::ShaderModule,
                       entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let pl_init = layout("init", &[Some(&bgl_params), Some(&bgl_init)]);
        let pl_nucleate = layout("nucleate", &[Some(&bgl_params), Some(&bgl_nucleate)]);
        let pl_prim = layout("primitives", &[Some(&bgl_params), Some(&bgl_prim)]);
        let pl_step = layout("step", &[Some(&bgl_params_stage), Some(&bgl_step)]);
        let pl_vis = layout("vis", &[Some(&bgl_params), Some(&bgl_vis)]);
        let pl_reduce = layout("reduce", &[Some(&bgl_params), Some(&bgl_reduce)]);

        let pipe_init = compute("init", &pl_init, &m_init, "main");
        let pipe_nucleate = compute("nucleate", &pl_nucleate, &m_nucleate, "main");
        let pipe_prim = compute("primitives", &pl_prim, &m_prim, "main");
        let pipe_step = compute("step", &pl_step, &m_step, "main");
        let pipe_vis = compute("vis", &pl_vis, &m_vis, "main");
        let pipe_reduce1 = compute("reduce pass 1", &pl_reduce, &m_reduce, "reduce_pass1");
        let pipe_reduce2 = compute("reduce pass 2", &pl_reduce, &m_reduce, "reduce_pass2");

        // ---- bind groups ------------------------------------------------------
        let bg_params = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("params"),
            layout: &bgl_params,
            entries: &[entry(0, &params_ub)],
        });
        let bg_params_stage = std::array::from_fn(|s| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("params + stage"),
                layout: &bgl_params_stage,
                entries: &[entry(0, &params_ub), entry(1, &stage_ub[s])],
            })
        });
        let bg_init = std::array::from_fn(|k| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("init"),
                layout: &bgl_init,
                entries: &[entry(0, &field[k]), entry(1, &fluid[k])],
            })
        });
        let bg_nucleate = std::array::from_fn(|k| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("nucleate"),
                layout: &bgl_nucleate,
                entries: &[entry(0, &field[k]), entry(1, &bubbles), entry(2, &batch_ub)],
            })
        });
        let bg_prim = std::array::from_fn(|k| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("primitives"),
                layout: &bgl_prim,
                entries: &[entry(0, &field[k]), entry(1, &fluid[k]), entry(2, &prim)],
            })
        });
        let bg_vis = std::array::from_fn(|k| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vis"),
                layout: &bgl_vis,
                entries: &[
                    entry(0, &field[k]),
                    entry(1, &fluid[k]),
                    entry(2, &prim),
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&vis_view),
                    },
                ],
            })
        });
        let bg_reduce = std::array::from_fn(|k| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("reduce"),
                layout: &bgl_reduce,
                entries: &[
                    entry(0, &field[k]),
                    entry(1, &fluid[k]),
                    entry(2, &prim),
                    entry(3, &partials),
                    entry(4, &diag),
                ],
            })
        });

        // `bg_step[cur][stage]`, matching the slot rotation in `record_step`.
        let bg_step = std::array::from_fn(|cur| {
            let slots = stage_slots(cur);
            std::array::from_fn(|stage| {
                let (in_slot, out_slot) = slots[stage];
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("step"),
                    layout: &bgl_step,
                    entries: &[
                        entry(0, &field[cur]),
                        entry(1, &fluid[cur]),
                        entry(2, &field[in_slot]),
                        entry(3, &fluid[in_slot]),
                        entry(4, &prim),
                        entry(5, &field[out_slot]),
                        entry(6, &fluid[out_slot]),
                    ],
                })
            })
        });

        let schedule = make_schedule(
            spec.nucleation,
            spec.bubbles,
            model.seed_radius(),
            grid,
            spec.nucleation_duration,
            spec.seed,
        );

        let mut sim = Self {
            model,
            grid,
            spec,
            time: 0.0,
            steps: 0,
            vis_gain: 1.0,
            n_cells,
            schedule,
            next_event: 0,
            nucleated_this_frame: false,
            field,
            fluid,
            prim,
            partials,
            diag,
            diag_staging,
            bubbles,
            batch_ub,
            params_ub,
            stage_ub,
            vis_view,
            _vis_tex: vis_tex,
            pipe_init,
            pipe_nucleate,
            pipe_prim,
            pipe_step,
            pipe_vis,
            pipe_reduce1,
            pipe_reduce2,
            bg_params,
            bg_params_stage,
            bg_init,
            bg_nucleate,
            bg_prim,
            bg_step,
            bg_vis,
            bg_reduce,
            cur: 0,
            readback_state: Arc::new(AtomicU32::new(ReadbackState::Idle as u32)),
            diagnostics: Diagnostics::default(),
            copy_issued: false,
        };

        sim.upload_stage_params(queue);
        sim.reset(device, queue, None);
        Ok(sim)
    }

    // -----------------------------------------------------------------------
    //  Parameters
    // -----------------------------------------------------------------------

    fn gpu_params(&self) -> SimParamsGpu {
        let m = &self.model;
        let e_ref = m.e_ref();
        SimParamsGpu {
            nx: self.grid[0],
            ny: self.grid[1],
            nz: self.grid[2],
            _pad_u0: 0,
            dx: m.dx,
            inv_dx: 1.0 / m.dx,
            dt: m.dt(),
            eta: m.eta,
            lambda: m.lambda,
            eps: m.eps(),
            phi_b: m.phi_b,
            m2: m.m2(),
            delta: m.delta(),
            a_rad: m.a_rad(),
            wall_width: m.wall_width(),
            // Floors are relative to the reference state, so they scale with the
            // chosen parameters instead of being magic absolute numbers.
            p_floor: 1e-7 * e_ref,
            e_floor: 1e-7 * e_ref,
            z_max: 0.9999,
            e_ref,
            t_ref: m.t_ref(),
            time: self.time,
            vis_gain: self.vis_gain,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }

    fn upload_params(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.params_ub, 0, bytemuck::bytes_of(&self.gpu_params()));
    }

    fn upload_stage_params(&self, queue: &wgpu::Queue) {
        let dt = self.model.dt();
        for (s, (a0, a1)) in RK3.iter().enumerate() {
            let p = StageParamsGpu { a0: *a0, a1: *a1, dt_stage: dt, _pad: 0.0 };
            queue.write_buffer(&self.stage_ub[s], 0, bytemuck::bytes_of(&p));
        }
    }

    /// Apply a live parameter change from the UI.  Cheap: nothing is
    /// reallocated, the next dispatch simply reads different constants.
    ///
    /// Note that changing the potential mid-run changes the total energy of the
    /// configuration, so the conservation readout will step.
    pub fn set_model(&mut self, queue: &wgpu::Queue, model: Model) {
        if model.validate().is_err() {
            return;
        }
        self.model = model;
        self.upload_stage_params(queue);
        self.upload_params(queue);
    }

    pub fn dispatch_dims(&self) -> (u32, u32, u32) {
        (self.grid[0].div_ceil(WG_X), self.grid[1], self.grid[2])
    }

    pub fn cell_count(&self) -> u64 {
        self.n_cells
    }

    pub fn remaining_bubbles(&self) -> usize {
        self.schedule.len().saturating_sub(self.next_event)
    }

    /// Device memory this simulation actually holds, counted from the
    /// allocations themselves rather than predicted.
    ///
    /// [`Self::lattice_bytes`] is the estimate used *before* allocating; this
    /// is the truth afterwards. They agree to well under a percent -- the
    /// difference is the fixed-size scratch (reduction partials, diagnostics,
    /// the bubble list and the parameter uniforms), a few tens of kilobytes.
    pub fn device_memory_bytes(&self) -> u64 {
        let buffers: u64 = self
            .field
            .iter()
            .chain(self.fluid.iter())
            .chain([&self.prim, &self.partials, &self.diag, &self.bubbles, &self.batch_ub])
            .chain(self.stage_ub.iter())
            .chain([&self.params_ub])
            .map(wgpu::Buffer::size)
            .sum();
        // The visualisation texture is rgba16float: 8 bytes per cell.
        buffers + self.n_cells * 8
    }

    /// Device memory a lattice of the given size will need, in bytes.
    ///
    /// Three RK state slots (`field` 8 B + `fluid` 16 B per cell), the
    /// primitive cache (16 B), and the visualisation texture (`rgba16float`,
    /// 8 B) -- 96 bytes per cell. Everything else (parameter uniforms, the
    /// 1024-entry reduction scratch, the bubble list) is a few tens of
    /// kilobytes and is ignored here.
    pub fn lattice_bytes(grid: [u32; 3]) -> u64 {
        let cells = grid[0] as u64 * grid[1] as u64 * grid[2] as u64;
        cells * (3 * (8 + 16) + 16 + 8)
    }

    /// The largest cubic lattice that fits both the device's binding limit and
    /// a memory budget, and which of the two binds.
    ///
    /// Two independent ceilings apply. `max_storage_buffer_binding_size` caps
    /// the `fluid` buffer at 16 bytes per cell and no amount of free memory
    /// relaxes it; the memory budget caps the total at 96 bytes per cell. Pass
    /// `None` for the budget when the backend cannot report memory, in which
    /// case only the binding limit applies.
    pub fn max_cubic_size(max_storage_binding: u64, budget_bytes: Option<u64>) -> LatticeCap {
        let side = |cells: u64| (cells as f64).cbrt().floor() as u32;
        let by_binding = side(max_storage_binding / 16);
        let by_budget = budget_bytes.map_or(u32::MAX, |b| side(b / 96));
        LatticeCap {
            side: by_binding.min(by_budget).clamp(MIN_GRID, MAX_GRID),
            limited_by_memory: by_budget < by_binding,
        }
    }

    /// Rebuild at a new lattice size, keeping every other parameter.
    ///
    /// The old lattice is released *before* the new one is allocated, so peak
    /// device memory is `max(old, new)` rather than `old + new`. Without that,
    /// shrinking a lattice could fail for want of memory, which would be an
    /// absurd way to run out. The intermediate is a minimum-size simulation,
    /// which costs one extra pipeline build (a few milliseconds, on a
    /// user-initiated action) and keeps `self` valid throughout.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grid: [u32; 3],
    ) -> Result<()> {
        let spec = SimulationSpec { grid, ..self.spec };
        // Validate before releasing anything, so a rejected size leaves the
        // running simulation untouched.
        Self::check_fits(device, grid)?;

        // Let in-flight work finish; wgpu cannot reclaim memory still
        // referenced by a queued submission.
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        *self = Self::new(device, queue, SimulationSpec { grid: [MIN_GRID; 3], ..spec })?;
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        *self = Self::new(device, queue, spec)?;
        Ok(())
    }

    fn check_fits(device: &wgpu::Device, grid: [u32; 3]) -> Result<()> {
        for (axis, n) in grid.iter().enumerate() {
            anyhow::ensure!(
                *n >= MIN_GRID,
                "grid axis {axis} is {n}; the 5-point stencil needs at least {MIN_GRID} cells"
            );
        }
        let limits = device.limits();
        let cells = grid[0] as u64 * grid[1] as u64 * grid[2] as u64;
        anyhow::ensure!(
            cells * 16 <= limits.max_storage_buffer_binding_size,
            "a {}x{}x{} lattice needs {:.2} GB per fluid buffer, above this device's \
             max_storage_buffer_binding_size of {:.2} GB",
            grid[0], grid[1], grid[2],
            cells as f64 * 16.0 / 1e9,
            limits.max_storage_buffer_binding_size as f64 / 1e9,
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    //  Lifecycle
    // -----------------------------------------------------------------------

    /// Reinitialise to the homogeneous symmetric phase and rebuild the
    /// nucleation schedule.
    pub fn reset(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        new_schedule: Option<Vec<BubbleEvent>>,
    ) {
        self.time = 0.0;
        self.steps = 0;
        self.cur = 0;
        self.next_event = 0;
        if let Some(s) = new_schedule {
            self.schedule = s;
        }
        self.upload_params(queue);
        self.upload_stage_params(queue);

        let (gx, gy, gz) = self.dispatch_dims();
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("reset") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("init"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_init);
            pass.set_bind_group(0, &self.bg_params, &[]);
            pass.set_bind_group(1, &self.bg_init[self.cur], &[]);
            pass.dispatch_workgroups(gx, gy, gz);
        }
        self.record_visualisation(&mut encoder);
        queue.submit(Some(encoder.finish()));
    }

    // -----------------------------------------------------------------------
    //  Time stepping
    // -----------------------------------------------------------------------

    /// Record `n_steps` SSP-RK3 time steps, plus any nucleation events that
    /// have come due, into `encoder`.
    pub fn record_steps(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        n_steps: u32,
    ) {
        self.nucleated_this_frame = false;
        self.upload_params(queue);
        self.record_nucleation(queue, encoder);
        for _ in 0..n_steps {
            self.record_step(encoder);
        }
    }

    fn record_nucleation(&mut self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        let mut batch: Vec<BubbleGpu> = Vec::new();
        while self.next_event < self.schedule.len()
            && self.schedule[self.next_event].time <= self.time
            && batch.len() < MAX_BUBBLES
        {
            let e = &self.schedule[self.next_event];
            batch.push(BubbleGpu {
                cx: e.pos[0],
                cy: e.pos[1],
                cz: e.pos[2],
                r0: e.radius,
            });
            self.next_event += 1;
        }
        if batch.is_empty() {
            return;
        }

        queue.write_buffer(&self.bubbles, 0, bytemuck::cast_slice(&batch));
        queue.write_buffer(
            &self.batch_ub,
            0,
            bytemuck::bytes_of(&NucleationBatchGpu { count: batch.len() as u32, _pad: [0; 3] }),
        );

        let (gx, gy, gz) = self.dispatch_dims();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("nucleate"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipe_nucleate);
        pass.set_bind_group(0, &self.bg_params, &[]);
        pass.set_bind_group(1, &self.bg_nucleate[self.cur], &[]);
        pass.dispatch_workgroups(gx, gy, gz);
        self.nucleated_this_frame = true;
    }

    fn record_step(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let (gx, gy, gz) = self.dispatch_dims();
        let slots = stage_slots(self.cur);

        for (stage, (in_slot, _out_slot)) in slots.into_iter().enumerate() {

            // Primitive recovery for this stage's input state.
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("primitives"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipe_prim);
                pass.set_bind_group(0, &self.bg_params, &[]);
                pass.set_bind_group(1, &self.bg_prim[in_slot], &[]);
                pass.dispatch_workgroups(gx, gy, gz);
            }

            // Fluxes, sources, and the RK stage combination.
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("step"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipe_step);
                pass.set_bind_group(0, &self.bg_params_stage[stage], &[]);
                pass.set_bind_group(1, &self.bg_step[self.cur][stage], &[]);
                pass.dispatch_workgroups(gx, gy, gz);
            }
        }

        // The final stage wrote into slot (cur + 1) % 3; see `stage_slots`.
        self.cur = (self.cur + 1) % 3;
        self.time += self.model.dt();
        self.steps += 1;
    }

    /// Refresh the primitives and the visualisation texture for the current
    /// state.  Must run after the last step of a frame, before rendering.
    pub fn record_visualisation(&self, encoder: &mut wgpu::CommandEncoder) {
        let (gx, gy, gz) = self.dispatch_dims();
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("primitives (vis)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_prim);
            pass.set_bind_group(0, &self.bg_params, &[]);
            pass.set_bind_group(1, &self.bg_prim[self.cur], &[]);
            pass.dispatch_workgroups(gx, gy, gz);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vis"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_vis);
            pass.set_bind_group(0, &self.bg_params, &[]);
            pass.set_bind_group(1, &self.bg_vis[self.cur], &[]);
            pass.dispatch_workgroups(gx, gy, gz);
        }
    }

    // -----------------------------------------------------------------------
    //  Diagnostics
    // -----------------------------------------------------------------------

    /// Record the global reduction and a copy into the staging buffer, but only
    /// if the previous readback has been consumed.  Assumes
    /// [`Self::record_visualisation`] already refreshed `prim` for this state.
    pub fn record_diagnostics(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.readback_state.load(Ordering::Acquire) != ReadbackState::Idle as u32 {
            return;
        }
        encoder.clear_buffer(&self.diag, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reduce"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_reduce1);
            pass.set_bind_group(0, &self.bg_params, &[]);
            pass.set_bind_group(1, &self.bg_reduce[self.cur], &[]);
            pass.dispatch_workgroups(N_PARTIALS, 1, 1);

            pass.set_pipeline(&self.pipe_reduce2);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.diag, 0, &self.diag_staging, 0, DIAG_BYTES);
        self.copy_issued = true;
        let _ = REDUCE_WG; // documented invariant, enforced by the shader constant
    }

    /// Kick off the asynchronous map.  Call once, immediately after submitting
    /// the encoder that [`Self::record_diagnostics`] wrote into.
    pub fn after_submit(&mut self) {
        if !self.copy_issued {
            return;
        }
        self.copy_issued = false;
        self.readback_state
            .store(ReadbackState::Pending as u32, Ordering::Release);
        let state = self.readback_state.clone();
        self.diag_staging.slice(..).map_async(wgpu::MapMode::Read, move |res| {
            let next = if res.is_ok() { ReadbackState::Ready } else { ReadbackState::Idle };
            state.store(next as u32, Ordering::Release);
        });
    }

    /// Consume a completed readback, if one is ready.  Never blocks.
    pub fn poll_diagnostics(&mut self) -> Diagnostics {
        if self.readback_state.load(Ordering::Acquire) == ReadbackState::Ready as u32 {
            {
                let view = self.diag_staging.slice(..).get_mapped_range();
                let raw: DiagnosticsGpu = *bytemuck::from_bytes(&view[..DIAG_BYTES as usize]);
                let inv_n = 1.0 / self.n_cells as f32;
                self.diagnostics = Diagnostics {
                    mean_energy: raw.sums[0] * inv_n,
                    broken_fraction: raw.sums[1] * inv_n,
                    mean_kinetic: raw.sums[2] * inv_n,
                    mean_fluid_energy: raw.sums[3] * inv_n,
                    max_velocity: f32::from_bits(raw.max_v_bits),
                };
            }
            self.diag_staging.unmap();
            self.readback_state
                .store(ReadbackState::Idle as u32, Ordering::Release);
        }
        self.diagnostics
    }

    /// Queue an extra bubble at a random position, effective immediately.
    pub fn nucleate_now(&mut self, rng: &mut impl rand::Rng) {
        let r = self.model.seed_radius();
        let event = BubbleEvent {
            time: self.time,
            pos: [
                rng.random::<f32>() * self.grid[0] as f32,
                rng.random::<f32>() * self.grid[1] as f32,
                rng.random::<f32>() * self.grid[2] as f32,
            ],
            radius: r,
        };
        self.schedule.insert(self.next_event, event);
    }
}

/// Whole-buffer bind group entry.  A free function rather than a closure so the
/// borrow of `buf` can flow into the returned entry's lifetime.
fn entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: buf.as_entire_binding() }
}

/// Slot rotation for SSP-RK3 with three state buffers.
///
/// With `a = cur`, `b = (cur+1)%3`, `c = (cur+2)%3`:
///
/// | stage | reads `U^n` | reads `U^k` | writes |
/// |-------|-------------|-------------|--------|
/// | 0     | a           | a           | b      |
/// | 1     | a           | b           | c      |
/// | 2     | a           | c           | b      |
///
/// so the result lands in `b`, which becomes the new `cur`.  No stage writes a
/// slot it also reads.
fn stage_slots(cur: usize) -> [(usize, usize); 3] {
    let a = cur;
    let b = (cur + 1) % 3;
    let c = (cur + 2) % 3;
    [(a, b), (b, c), (c, b)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_slots_never_alias_output_with_input() {
        for cur in 0..3 {
            let slots = stage_slots(cur);
            for (stage, (in_slot, out_slot)) in slots.iter().enumerate() {
                assert_ne!(in_slot, out_slot, "stage {stage} of cur {cur} writes its input");
                assert_ne!(cur, *out_slot, "stage {stage} of cur {cur} overwrites U^n");
            }
            // The final stage must land in the slot that becomes the new `cur`.
            assert_eq!(slots[2].1, (cur + 1) % 3);
        }
    }

    #[test]
    fn parameter_blocks_have_the_expected_size() {
        // Must match the WGSL struct layouts (std140-style uniform rules: all
        // members here are 4-byte scalars, so the layout is dense).
        assert_eq!(std::mem::size_of::<SimParamsGpu>(), 96);
        assert_eq!(std::mem::size_of::<StageParamsGpu>(), 16);
        assert_eq!(std::mem::size_of::<BubbleGpu>(), 16);
        assert_eq!(std::mem::size_of::<DiagnosticsGpu>() as u64, DIAG_BYTES);
    }

    #[test]
    fn lattice_bytes_is_96_per_cell() {
        assert_eq!(Simulation::lattice_bytes([100, 10, 10]), 10_000 * 96);
    }

    #[test]
    fn cap_reports_which_limit_binds() {
        // 2 GB binding limit, plenty of memory -> the binding limit binds.
        let plenty = Simulation::max_cubic_size(2_147_483_648, Some(1 << 60));
        assert!(!plenty.limited_by_memory);
        assert_eq!(plenty.side, ((2_147_483_648u64 / 16) as f64).cbrt().floor() as u32);

        // Same device, 1 GB of memory -> memory binds.
        let tight = Simulation::max_cubic_size(2_147_483_648, Some(1_000_000_000));
        assert!(tight.limited_by_memory);
        assert!(tight.side < plenty.side);
        assert!(Simulation::lattice_bytes([tight.side; 3]) <= 1_000_000_000);

        // No memory report -> only the binding limit applies.
        assert_eq!(Simulation::max_cubic_size(2_147_483_648, None).side, plenty.side);
    }

    #[test]
    fn cap_never_goes_below_the_stencil_minimum() {
        let cap = Simulation::max_cubic_size(1024, Some(1024));
        assert_eq!(cap.side, MIN_GRID);
    }

    #[test]
    fn rk3_coefficients_are_consistent() {
        // Each stage is a convex combination, which is what makes SSP-RK3
        // strong-stability-preserving.
        for (a0, a1) in RK3 {
            assert!((a0 + a1 - 1.0).abs() < 1e-6);
            assert!(a0 >= 0.0 && a1 > 0.0);
        }
    }
}
