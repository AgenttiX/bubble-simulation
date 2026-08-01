//! Offscreen rendering: draw one frame to a PNG without a window.
//!
//! Useful for producing figures for teaching material, and it is how the
//! renderer is checked on a machine with no display (or a locked screen) --
//! which is otherwise a blind spot, since every other automated check only
//! exercises the solver.
//!
//! This is the one place in the program that deliberately copies GPU memory
//! back to the host, and it copies the *rendered image*, never the lattice.

use anyhow::{Context, Result};

use crate::camera::OrbitCamera;
use crate::gpu::render::{RenderSettings, VolumeRenderer};

/// Buffer rows in a texture-to-buffer copy must start on a 256-byte boundary.
const ROW_ALIGN: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

pub fn render_to_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    vis_view: &wgpu::TextureView,
    grid: [u32; 3],
    settings: &RenderSettings,
    size: (u32, u32),
    path: &std::path::Path,
) -> Result<()> {
    let (width, height) = size;
    let format = wgpu::TextureFormat::Rgba8Unorm;

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen target"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut renderer = VolumeRenderer::new(device, format, vis_view, grid);
    let mut camera = OrbitCamera::default();
    camera.frame_box(renderer.box_half());
    renderer.update(
        queue,
        camera.view_proj(width as f32 / height as f32),
        camera.eye(),
        settings,
    );

    let bytes_per_row = (width * 4).div_ceil(ROW_ALIGN) * ROW_ALIGN;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("screenshot staging"),
        size: (bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("capture") });
    renderer.draw(&mut encoder, &target_view);
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    queue.submit(Some(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .context("device poll failed")?;
    rx.recv()
        .context("readback channel closed")?
        .context("failed to map the screenshot staging buffer")?;

    // Drop the padding each row carries for the 256-byte alignment.
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    {
        let view = staging.slice(..).get_mapped_range();
        for row in 0..height {
            let start = (row * bytes_per_row) as usize;
            pixels.extend_from_slice(&view[start..start + (width * 4) as usize]);
        }
    }
    staging.unmap();

    let file = std::fs::File::create(path)
        .with_context(|| format!("cannot create {}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .context("failed to write the PNG header")?
        .write_image_data(&pixels)
        .context("failed to write the PNG data")?;

    println!("wrote {} ({width}x{height})", path.display());
    Ok(())
}
