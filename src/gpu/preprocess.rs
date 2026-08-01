//! A minimal WGSL assembler: `//!include <name>` resolution plus injection of
//! the floating-point precision alias.
//!
//! WGSL has no preprocessor and no module system, so the shaders are stitched
//! together here.  Two rules:
//!
//!   * `//!include foo` splices in `shaders/foo.wgsl`, recursively, and each
//!     module is spliced at most once per translation unit.
//!   * A precision prologue defining `alias flt = ...` is prepended to every
//!     module.  All physics shaders spell their types `flt` rather than `f32`,
//!     which is what makes the width a single-point decision on the shader
//!     side.  See `docs/PRECISION.md` for what else changing it entails.

use std::collections::HashSet;

/// Floating point width used by the *simulation* shaders.  Rendering is always
/// `f32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Precision {
    #[default]
    Single,
    /// Not reachable today: WGSL (and therefore wgpu) has no 64-bit float type.
    /// Kept as an explicit marker of where the decision lives; see
    /// `docs/PRECISION.md` for the migration paths.
    #[allow(dead_code)]
    Double,
}

impl Precision {
    fn prologue(self) -> &'static str {
        match self {
            Precision::Single => "alias flt = f32;\n",
            // A WGSL implementation with a 64-bit float extension would need
            // its own `enable` directive here, e.g. `enable f64;`.
            Precision::Double => {
                panic!(
                    "double precision is not available through WGSL/wgpu; \
                     see docs/PRECISION.md for the supported migration paths"
                )
            }
        }
    }
}

/// Shader sources, embedded at build time so the binary is self-contained.
const MODULES: &[(&str, &str)] = &[
    ("common", include_str!("../../shaders/common.wgsl")),
    ("potential", include_str!("../../shaders/potential.wgsl")),
    ("eos", include_str!("../../shaders/eos.wgsl")),
    ("primitives", include_str!("../../shaders/primitives.wgsl")),
    ("step", include_str!("../../shaders/step.wgsl")),
    ("init", include_str!("../../shaders/init.wgsl")),
    ("nucleate", include_str!("../../shaders/nucleate.wgsl")),
    ("vis", include_str!("../../shaders/vis.wgsl")),
    ("reduce", include_str!("../../shaders/reduce.wgsl")),
    ("volume", include_str!("../../shaders/volume.wgsl")),
];

fn source_of(name: &str) -> &'static str {
    MODULES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| *s)
        .unwrap_or_else(|| panic!("unknown shader module `{name}`"))
}

/// Assemble a simulation shader, with includes resolved and the precision
/// alias in place.
pub fn build(entry_module: &str, precision: Precision) -> String {
    let mut out = String::from(precision.prologue());
    let mut seen = HashSet::new();
    seen.insert(entry_module.to_string());
    splice(entry_module, &mut out, &mut seen);
    out
}

/// Assemble a rendering shader.  These are unconditionally `f32`, but they may
/// still use includes.
pub fn build_render(entry_module: &str) -> String {
    let mut out = String::new();
    let mut seen = HashSet::new();
    seen.insert(entry_module.to_string());
    splice(entry_module, &mut out, &mut seen);
    out
}

fn splice(name: &str, out: &mut String, seen: &mut HashSet<String>) {
    for line in source_of(name).lines() {
        match line.trim().strip_prefix("//!include") {
            Some(rest) => {
                let dep = rest.trim();
                if seen.insert(dep.to_string()) {
                    splice(dep, out, seen);
                }
            }
            None => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn includes_are_resolved_once() {
        // Both `step` and its include `eos` pull in `common`.
        let src = build("step", Precision::Single);
        assert_eq!(src.matches("struct SimParams").count(), 1);
        assert!(src.contains("alias flt = f32;"));
        assert!(src.contains("fn eos_wave_speeds"));
        assert!(src.contains("fn dpotential"));
        assert!(!src.contains("//!include"));
    }

    #[test]
    fn every_module_assembles() {
        for (name, _) in MODULES {
            let src = if *name == "volume" {
                build_render(name)
            } else {
                build(name, Precision::Single)
            };
            assert!(!src.contains("//!include"), "{name} has unresolved includes");
        }
    }
}
