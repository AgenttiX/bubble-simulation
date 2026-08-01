//! Volume renderer.
//!
//! One full-screen triangle; the fragment shader ray-marches the 3D texture
//! that the simulation's `vis` pass just wrote.  There is no vertex data, no
//! mesh extraction, and no transfer of lattice state to the host - the frame is
//! produced from the same device memory the solver is writing.
//!
//! Colour management: the swapchain is deliberately configured with a *non*-
//! sRGB format, so this shader emits gamma-encoded values directly (its colour
//! maps are specified in sRGB space) and egui uses its matching gamma path.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::gpu::preprocess;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum FieldMode {
    /// The order parameter itself: which regions have converted.
    Phi = 0,
    /// Magnitude of the fluid three-velocity.
    Speed = 1,
    /// Signed temperature contrast; shows compression and rarefaction.
    Temperature = 2,
    /// Fluid kinetic energy density -- what sources gravitational waves.
    #[default]
    Kinetic = 3,
}

impl FieldMode {
    pub const ALL: [FieldMode; 4] = [
        FieldMode::Phi,
        FieldMode::Speed,
        FieldMode::Temperature,
        FieldMode::Kinetic,
    ];

    pub fn label(self) -> &'static str {
        match self {
            FieldMode::Phi => "Order parameter  phi",
            FieldMode::Speed => "Fluid speed  |v|",
            FieldMode::Temperature => "Temperature contrast  dT/T",
            FieldMode::Kinetic => "Kinetic energy  w W^2 v^2",
        }
    }

    /// A gain that puts each field roughly in range for a fresh simulation.
    pub fn default_gain(self) -> f32 {
        match self {
            // Chosen so a strong feature reaches an optical depth of a few
            // across the box at the default `absorption`, leaving the interior
            // legible rather than blocked out.
            FieldMode::Phi => 0.5,
            FieldMode::Speed => 4.0,
            FieldMode::Temperature => 15.0,
            FieldMode::Kinetic => 20.0,
        }
    }

    /// Signed fields want the signed ramp; unsigned ones do not.
    pub fn default_colormap(self) -> usize {
        match self {
            FieldMode::Phi => 0,
            FieldMode::Temperature => 2,
            _ => 1,
        }
    }
}

pub const COLORMAPS: [&str; 3] = ["viridis", "inferno", "signed"];

#[derive(Clone, Copy, Debug)]
pub struct RenderSettings {
    pub field_mode: FieldMode,
    pub colormap: usize,
    /// Order-parameter value, in units of `phi_b`, taken to be the wall.
    pub iso_level: f32,
    pub wall_opacity: f32,
    /// Ray-march sample density.  1 means one sample per lattice cell.
    pub samples_per_cell: f32,
    pub exposure: f32,
    pub field_gain: f32,
    /// Optical depth across the whole box at unit field intensity.  Independent
    /// of the lattice size, so the look survives a change of `--grid`.
    pub absorption: f32,
    pub show_iso: bool,
    pub show_volume: bool,
    pub show_box: bool,
    pub background: f32,
    /// -1 disables the cutaway; 0/1/2 select the x/y/z half-space.
    pub clip_axis: i32,
    /// Cutaway plane position in [-1, 1] along the chosen axis.
    pub clip_pos: f32,
}

impl RenderSettings {
    pub fn with_field(field_mode: FieldMode) -> Self {
        Self {
            field_mode,
            colormap: field_mode.default_colormap(),
            field_gain: field_mode.default_gain(),
            ..Self::default()
        }
    }
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            field_mode: FieldMode::default(),
            colormap: FieldMode::default().default_colormap(),
            iso_level: 0.5,
            wall_opacity: 0.55,
            samples_per_cell: 1.5,
            exposure: 1.5,
            field_gain: FieldMode::default().default_gain(),
            absorption: 12.0,
            show_iso: true,
            show_volume: true,
            show_box: true,
            background: 1.0,
            clip_axis: -1,
            clip_pos: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ViewUniform {
    inv_view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    box_half: [f32; 4],
    grid: [f32; 4],
    render0: [f32; 4],
    render1: [f32; 4],
    render2: [f32; 4],
    render3: [f32; 4],
}

pub struct VolumeRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    /// Half-extents of the lattice in world units, normalised so the longest
    /// axis spans 1.0.
    box_half: Vec3,
    /// World-space size of one cell.
    cell: f32,
    grid: [u32; 3],
    frame: u32,
}

impl VolumeRenderer {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        vis_view: &wgpu::TextureView,
        grid: [u32; 3],
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volume"),
            source: wgpu::ShaderSource::Wgsl(preprocess::build_render("volume").into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume view"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view uniform"),
            size: std::mem::size_of::<ViewUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Clamp rather than repeat: the ray never leaves the box, and clamping
        // keeps the trilinear filter from wrapping across the periodic boundary
        // at the very outermost half-cell.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volume sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let bind_group = make_bind_group(device, &layout, &uniform, vis_view, &sampler);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volume"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volume"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let n_max = grid.iter().copied().max().unwrap_or(1) as f32;
        let box_half = Vec3::new(
            0.5 * grid[0] as f32 / n_max,
            0.5 * grid[1] as f32 / n_max,
            0.5 * grid[2] as f32 / n_max,
        );

        Self {
            pipeline,
            bind_group,
            uniform,
            box_half,
            cell: 1.0 / n_max,
            grid,
            frame: 0,
        }
    }

    pub fn box_half(&self) -> Vec3 {
        self.box_half
    }

    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        view_proj: Mat4,
        eye: Vec3,
        settings: &RenderSettings,
    ) {
        self.frame = self.frame.wrapping_add(1);
        let u = ViewUniform {
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            cam_pos: [eye.x, eye.y, eye.z, 0.0],
            box_half: [self.box_half.x, self.box_half.y, self.box_half.z, self.cell],
            grid: [
                self.grid[0] as f32,
                self.grid[1] as f32,
                self.grid[2] as f32,
                0.0,
            ],
            render0: [
                settings.iso_level,
                settings.wall_opacity,
                settings.samples_per_cell,
                settings.exposure,
            ],
            render1: [
                settings.field_mode as i32 as f32,
                settings.clip_axis as f32,
                settings.clip_pos,
                settings.colormap as f32,
            ],
            render2: [
                settings.field_gain,
                settings.absorption,
                if settings.show_iso { 1.0 } else { 0.0 },
                if settings.show_volume { 1.0 } else { 0.0 },
            ],
            render3: [
                // Animate the dither so banding averages away over time.
                (self.frame % 64) as f32 * 0.618_034,
                if settings.show_box { 1.0 } else { 0.0 },
                settings.background,
                0.0,
            ],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
    }

    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("volume"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn make_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    vis_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("volume view"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(vis_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_uniform_matches_the_wgsl_struct() {
        // mat4x4 (64) + 7 * vec4 (112).
        assert_eq!(std::mem::size_of::<ViewUniform>(), 176);
    }

    #[test]
    fn every_field_mode_has_a_valid_default_colormap() {
        for m in FieldMode::ALL {
            assert!(m.default_colormap() < COLORMAPS.len());
            assert!(m.default_gain() > 0.0);
        }
    }
}
