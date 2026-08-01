//! Orbit camera.
//!
//! Rotate around the box, pan the point you orbit, and dolly in until you are
//! inside it -- which is the interesting vantage point once the bubbles have
//! merged and the box is full of sound waves.

use glam::{Mat4, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    /// Rotation about the world +y axis, radians.
    pub yaw: f32,
    /// Elevation, radians, clamped just short of the poles to avoid gimbal flip.
    pub pitch: f32,
    pub fov_y: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 2.1,
            yaw: 0.6,
            pitch: 0.42,
            fov_y: 50f32.to_radians(),
            znear: 0.002,
            zfar: 100.0,
        }
    }
}

const PITCH_LIMIT: f32 = 1.5533; // ~89 degrees

impl OrbitCamera {
    pub fn forward(&self) -> Vec3 {
        // From the eye towards the target.
        -self.offset().normalize()
    }

    fn offset(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cp * sy, sp, cp * cy) * self.distance
    }

    pub fn eye(&self) -> Vec3 {
        self.target + self.offset()
    }

    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize_or_zero()
    }

    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize_or_zero()
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        // `perspective_rh` produces wgpu's [0, 1] depth range.
        let proj = Mat4::perspective_rh(self.fov_y, aspect.max(1e-3), self.znear, self.zfar);
        let view = Mat4::look_at_rh(self.eye(), self.target, Vec3::Y);
        proj * view
    }

    /// Mouse drag, in radians per pixel-ish units.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx;
        self.pitch = (self.pitch + dy).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// Slide the orbit target in the camera plane.  Scaled by distance so the
    /// pan feels the same at every zoom level.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let scale = self.distance * 0.9;
        self.target += self.right() * (-dx * scale) + self.up() * (dy * scale);
    }

    /// Multiplicative zoom; `amount` is typically the scroll delta.
    pub fn zoom(&mut self, amount: f32) {
        self.distance = (self.distance * (1.0 - amount * 0.12)).clamp(0.02, 40.0);
    }

    /// Move the eye along the view direction, keeping the orientation.  Unlike
    /// `zoom` this pushes the target too, so you can fly through the volume.
    pub fn dolly(&mut self, amount: f32) {
        self.target += self.forward() * amount * self.distance;
    }

    pub fn frame_box(&mut self, half_extent: Vec3) {
        self.target = Vec3::ZERO;
        self.distance = half_extent.length() * 2.4;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eye_sits_at_the_requested_distance() {
        let c = OrbitCamera::default();
        assert!((c.eye().distance(c.target) - c.distance).abs() < 1e-5);
    }

    #[test]
    fn pitch_is_clamped_short_of_the_pole() {
        let mut c = OrbitCamera::default();
        c.orbit(0.0, 100.0);
        assert!(c.pitch <= PITCH_LIMIT);
        c.orbit(0.0, -200.0);
        assert!(c.pitch >= -PITCH_LIMIT);
        // The basis must stay well conditioned at the limit.
        assert!(c.right().length() > 0.5);
        assert!(c.up().length() > 0.5);
    }

    #[test]
    fn zoom_stays_positive() {
        let mut c = OrbitCamera::default();
        for _ in 0..200 {
            c.zoom(5.0);
        }
        assert!(c.distance > 0.0);
    }

    #[test]
    fn projection_maps_the_target_to_the_screen_centre() {
        let c = OrbitCamera::default();
        let clip = c.view_proj(1.6) * c.target.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        assert!(ndc.x.abs() < 1e-4 && ndc.y.abs() < 1e-4);
    }
}
