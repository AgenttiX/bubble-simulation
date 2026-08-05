//! Sets the `vulkan_vram` cfg on platforms where wgpu's Vulkan backend is
//! compiled in, which is where `src/gpu/vram.rs` can query the device memory
//! budget through the HAL.
//!
//! wgpu sets its own internal `vulkan` cfg, but that is private to wgpu, so we
//! reproduce the platform test here rather than relying on it.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(vulkan_vram)");

    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let vulkan = arch != "wasm32"
        && matches!(
            os.as_str(),
            "linux" | "windows" | "android" | "freebsd" | "openbsd" | "netbsd" | "dragonfly"
        );
    if vulkan {
        println!("cargo::rustc-cfg=vulkan_vram");
    }
}
