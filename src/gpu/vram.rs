//! Device memory budget.
//!
//! wgpu has no portable "how much VRAM is there" call -- deliberately, since
//! WebGPU cannot expose it. But the lattice is the single biggest allocation
//! this program makes, and letting a user pick a size that cannot be allocated
//! is a device loss rather than an error message, so it is worth reaching past
//! the abstraction to find out.
//!
//! On Vulkan we query it directly through the HAL:
//!
//!   * **capacity** -- the sum of `DEVICE_LOCAL` heap sizes, from
//!     `vkGetPhysicalDeviceMemoryProperties`. This is the card's VRAM.
//!   * **budget and usage** -- from `VK_EXT_memory_budget`, which reports what
//!     the driver is willing to give *this process right now*, accounting for
//!     everything else on the system. On a desktop with a compositor and a
//!     browser running, the difference between this and capacity is easily a
//!     couple of gigabytes, so it is the number that actually matters.
//!
//! Everywhere else this returns `None` and the caller falls back to the limits
//! wgpu does expose, which are hard API caps rather than physical ones.

/// What is known about device memory. All figures in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VramInfo {
    /// Total device-local memory on the adapter.
    pub capacity: u64,
    /// What the driver will currently hand this process, if reported.
    pub budget: Option<u64>,
    /// What this process is currently using, if reported.
    pub used: Option<u64>,
}

impl VramInfo {
    /// Memory available for a new allocation right now.
    ///
    /// Falls back to capacity when `VK_EXT_memory_budget` is unavailable, which
    /// is optimistic -- but the caller keeps a safety margin on top, and an
    /// optimistic estimate is better than refusing to give any guidance.
    pub fn available(&self) -> u64 {
        match (self.budget, self.used) {
            (Some(budget), Some(used)) => budget.saturating_sub(used),
            _ => self.capacity,
        }
    }

    /// Whether the free-memory figure is measured rather than assumed.
    pub fn available_is_measured(&self) -> bool {
        self.budget.is_some() && self.used.is_some()
    }
}

/// How much of the free budget the lattice may claim.
///
/// The remainder has to cover the swapchain, the egui font atlas, the driver's
/// own bookkeeping and allocator fragmentation. 85% is empirical headroom, not
/// a derived number, and it is deliberately generous: overshooting here does
/// not produce an error, it produces a lost device.
pub const LATTICE_BUDGET_FRACTION: f64 = 0.85;

/// Bytes a new lattice may occupy, given what is free now and what the current
/// lattice hands back when it is released.
///
/// Adding `current_lattice` back is what makes *shrinking* always possible.
/// The memory the running simulation holds counts as used as far as the driver
/// is concerned, but [`crate::gpu::sim::Simulation::resize`] releases it before
/// allocating the replacement, so it is genuinely available to the new one.
pub fn lattice_budget(available: u64, current_lattice: u64) -> u64 {
    ((available.saturating_add(current_lattice) as f64) * LATTICE_BUDGET_FRACTION) as u64
}

/// Query device memory. Returns `None` on non-Vulkan backends.
pub fn query(instance: &wgpu::Instance, adapter: &wgpu::Adapter) -> Option<VramInfo> {
    #[cfg(not(vulkan_vram))]
    {
        let _ = (instance, adapter);
        None
    }

    #[cfg(vulkan_vram)]
    unsafe {
        vulkan_query(instance, adapter)
    }
}

#[cfg(vulkan_vram)]
unsafe fn vulkan_query(instance: &wgpu::Instance, adapter: &wgpu::Adapter) -> Option<VramInfo> {
    use wgpu::hal::api::Vulkan;

    // SAFETY: both handles are only read from, are not destroyed, and do not
    // outlive the guards they came from.
    let hal_instance = unsafe { instance.as_hal::<Vulkan>() }?;
    let hal_adapter = unsafe { adapter.as_hal::<Vulkan>() }?;

    let shared = hal_instance.shared_instance();
    let raw_instance = shared.raw_instance();
    let physical_device = hal_adapter.raw_physical_device();

    let props = unsafe { raw_instance.get_physical_device_memory_properties(physical_device) };

    // Vulkan reports memory as heaps; the device-local ones are the VRAM.
    // Integrated GPUs mark system memory device-local too, which is correct:
    // there the "VRAM" genuinely is a slice of system RAM.
    //
    // Take the *largest* device-local heap rather than summing them. Discrete
    // NVIDIA cards expose a second, small device-local heap (the ~256 MB
    // host-visible BAR window) which is a view onto the same physical memory,
    // so summing over-reports capacity by that much and double-counts its
    // usage. There is no way to detect aliasing through the Vulkan API, and
    // every device that matters here has one dominant VRAM pool.
    let mut primary = None;
    let mut capacity = 0u64;
    for i in 0..props.memory_heap_count as usize {
        let heap = props.memory_heaps[i];
        if heap.flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL) && heap.size > capacity {
            capacity = heap.size;
            primary = Some(i);
        }
    }
    let primary = primary?;

    // The budget query needs two things: a Vulkan 1.1 instance, because
    // `vkGetPhysicalDeviceMemoryProperties2` does not exist in 1.0, and a
    // physical device that advertises VK_EXT_memory_budget, because chaining
    // its struct otherwise is invalid usage that the validation layers will
    // (rightly) flag. Without both, capacity alone is still useful.
    let has_props2 = shared.instance_api_version() >= ash::vk::API_VERSION_1_1;
    let has_budget_ext = has_props2
        && unsafe { raw_instance.enumerate_device_extension_properties(physical_device) }
            .map(|exts| {
                exts.iter().any(|e| {
                    e.extension_name_as_c_str() == Ok(ash::ext::memory_budget::NAME)
                })
            })
            .unwrap_or(false);

    if !has_budget_ext {
        return Some(VramInfo { capacity, budget: None, used: None });
    }

    let mut budget_props = ash::vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut props2 =
        ash::vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget_props);
    unsafe {
        raw_instance.get_physical_device_memory_properties2(physical_device, &mut props2);
    }

    let budget = budget_props.heap_budget[primary];
    let used = budget_props.heap_usage[primary];

    Some(VramInfo {
        capacity,
        // A driver that reports a zero budget is not reporting one.
        budget: (budget > 0).then_some(budget),
        used: (budget > 0).then_some(used),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn available_prefers_the_measured_budget() {
        let info = VramInfo {
            capacity: 24_000_000_000,
            budget: Some(23_000_000_000),
            used: Some(3_000_000_000),
        };
        assert_eq!(info.available(), 20_000_000_000);
        assert!(info.available_is_measured());
    }

    #[test]
    fn available_falls_back_to_capacity_when_unreported() {
        let info = VramInfo { capacity: 8_000_000_000, budget: None, used: None };
        assert_eq!(info.available(), 8_000_000_000);
        assert!(!info.available_is_measured());
    }

    /// Shrinking must always be possible: the memory the running lattice holds
    /// is counted as used, but it is released before the new one is allocated.
    #[test]
    fn budget_credits_back_the_current_lattice() {
        // 1 GB free, but the running lattice is holding 10 GB.
        let budget = lattice_budget(1_000_000_000, 10_000_000_000);
        assert!(
            budget > 9_000_000_000,
            "a 10 GB lattice must be able to shrink to, say, 2 GB; got {budget}"
        );
    }

    #[test]
    fn budget_keeps_headroom() {
        assert_eq!(lattice_budget(1_000, 0), 850);
    }

    /// Drivers can report usage above budget under pressure; that must clamp to
    /// zero rather than wrapping around to a huge number.
    #[test]
    fn over_budget_saturates_at_zero() {
        let info = VramInfo {
            capacity: 8_000_000_000,
            budget: Some(7_000_000_000),
            used: Some(7_500_000_000),
        };
        assert_eq!(info.available(), 0);
    }
}
