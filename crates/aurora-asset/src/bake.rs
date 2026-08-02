//! Aurora's runtime model format: a baked file that is READ, not parsed.
//!
//! FBX and glTF are interchange formats. They are built to carry a scene between
//! authoring tools, which means a reader has to walk a node graph, resolve
//! references, rebuild index buffers, and untangle whatever the exporter felt
//! like emitting. That cost is paid on every load, in every run, forever.
//!
//! It is not small. Poly Souls' bailey is 105 distinct source files, and parsing
//! them is what a room costs: 2.2 GB of peak process memory against 69 MiB of
//! mesh and 88 MiB of texture actually uploaded. The art is not big. The parse is.
//!
//! So a baked file is laid out the way the engine wants it. Vertices and indices
//! are the exact bytes the GPU is given, copied in one `memcpy` rather than
//! rebuilt element by element; a clip's keys are a flat block of floats; a
//! skeleton is an array of joints. Reading one is a bounds check and a copy.
//!
//! # Why hand-rolled rather than serde
//!
//! Because the point is the LAYOUT. A derived encoding would serialise field by
//! field and give back exactly the walk this exists to avoid - the win here is
//! that `Vec<Vertex>` is `Pod`, so a whole mesh is one length and one slice, and
//! that only happens if the format says so.
//!
//! # Compatibility
//!
//! The magic and version are checked and a mismatch is an ERROR, never a
//! reinterpretation. A stale bake read as a current one is a model built out of
//! whatever the bytes happened to mean, which fails somewhere far from here.

use glam::{Mat4, Quat, Vec3};

use crate::mesh::{MeshData, Vertex};
use crate::model::{Channel, Clip, Interp, Joint, Model, Path, Primitive, RootMotion, Skeleton};

/// `AURM`, little-endian.
const MAGIC: u32 = 0x4d52_5541;

/// Bumped whenever the layout below changes in any way. There is no migration
/// and there should not be: a bake is derived from source art that is still
/// sitting there, so the answer to a version mismatch is to bake it again.
const VERSION: u32 = 2;

/// The conventional extension for a baked model.
pub const EXT: &str = "aurm";

// --- writing ---------------------------------------------------------------

struct W {
    out: Vec<u8>,
}

impl W {
    fn u32(&mut self, v: u32) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }
    fn bool(&mut self, v: bool) {
        self.out.push(v as u8);
    }
    fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.out.extend_from_slice(s.as_bytes());
    }
    /// A length-prefixed block of plain data, copied whole.
    fn pod<T: bytemuck::Pod>(&mut self, v: &[T]) {
        self.u32(v.len() as u32);
        self.out.extend_from_slice(bytemuck::cast_slice(v));
    }
    fn f32s(&mut self, v: &[f32]) {
        self.pod(v);
    }
    /// `None` is a leading 0 and nothing else, so an absent texture costs a byte.
    ///
    /// The pixels are re-ENCODED rather than stored raw. Everything else in this
    /// format is stored the way the engine wants it, and a texture is the one
    /// thing where that is the wrong trade: a 4096 x 4096 atlas is 64 MiB of
    /// RGBA and about a megabyte as PNG, and a Synty pack embeds one into every
    /// module it ships. Bakes of the castle pack came to 2.2 GB from 29 MiB of
    /// source, which is not a bake, it is a decompression.
    ///
    /// The decode on the way back in is what the source file cost anyway.
    fn opt_tex(&mut self, t: &Option<crate::model::Tex>) {
        let Some((px, w, h)) = t else {
            self.bool(false);
            return;
        };
        // An EMPTIED texture - the renderer drops decoded pixels once they are
        // on the GPU and keeps the entry - has nothing to encode and nothing to
        // say. Baking one would write a texture that is not there.
        if px.is_empty() || *w == 0 || *h == 0 {
            self.bool(false);
            return;
        }
        let mut png = Vec::new();
        let ok = image::RgbaImage::from_raw(*w, *h, px.clone()).is_some_and(|img| {
            img.write_to(
                &mut std::io::Cursor::new(&mut png),
                image::ImageFormat::Png,
            )
            .is_ok()
        });
        if !ok {
            // Loud, and then absent. A texture that will not re-encode is a
            // broken bake, and writing a truncated one would be worse.
            eprintln!("aurora: bake: could not encode a {w}x{h} embedded texture");
            self.bool(false);
            return;
        }
        self.bool(true);
        self.pod(&png);
    }
}

/// Encode `model` as a baked file.
pub fn write(model: &Model) -> Vec<u8> {
    let mut w = W {
        // Room for the geometry up front: the whole point is to avoid a walk,
        // and growing a buffer forty times on the way out is a walk.
        out: Vec::with_capacity(1 << 20),
    };
    w.u32(MAGIC);
    w.u32(VERSION);

    w.u32(model.primitives.len() as u32);
    for p in &model.primitives {
        w.pod(&p.mesh.vertices);
        w.pod(&p.mesh.indices);
        w.str(&p.material);
        for v in p.base_color {
            w.f32(v);
        }
        w.f32(p.metallic);
        w.f32(p.roughness);
        for v in p.emissive {
            w.f32(v);
        }
        w.opt_tex(&p.texture);
        w.opt_tex(&p.normal_tex);
        w.opt_tex(&p.mr_tex);
        w.opt_tex(&p.emissive_tex);
        w.bool(p.skinned);
    }

    match &model.skeleton {
        None => w.bool(false),
        Some(s) => {
            w.bool(true);
            w.u32(s.joints.len() as u32);
            for j in &s.joints {
                // `usize::MAX` is the root: a parent index and "no parent" have
                // to stay distinguishable, and 0 is a real joint.
                w.u32(j.parent.map(|p| p as u32).unwrap_or(u32::MAX));
                for v in j.inverse_bind.to_cols_array() {
                    w.f32(v);
                }
                for v in j.t.to_array() {
                    w.f32(v);
                }
                for v in j.r.to_array() {
                    w.f32(v);
                }
                for v in j.s.to_array() {
                    w.f32(v);
                }
                w.str(&j.name);
            }
        }
    }

    w.u32(model.clips.len() as u32);
    for c in &model.clips {
        w.str(&c.name);
        w.f32(c.duration);
        w.u32(c.channels.len() as u32);
        for ch in &c.channels {
            w.u32(ch.joint as u32);
            w.u32(match ch.path {
                Path::Translation => 0,
                Path::Rotation => 1,
                Path::Scale => 2,
            });
            w.u32(match ch.interp {
                Interp::Linear => 0,
                Interp::Step => 1,
            });
            w.f32s(&ch.times);
            w.f32s(&ch.values);
        }
        match &c.root {
            None => w.bool(false),
            Some(r) => {
                w.bool(true);
                w.u32(match r.interp {
                    Interp::Linear => 0,
                    Interp::Step => 1,
                });
                w.f32s(&r.times);
                w.f32s(&r.values);
            }
        }
    }
    w.out
}

// --- reading ---------------------------------------------------------------

struct R<'a> {
    b: &'a [u8],
    at: usize,
}

/// Every read is bounds-checked and every failure names the offset.
///
/// A truncated or corrupt bake must be an error, not a panic and not a model
/// made of garbage: this file is generated, so the interesting case is a stale
/// or half-written one, and both look like plausible bytes.
impl<'a> R<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(n).ok_or("baked model: length overflow")?;
        if end > self.b.len() {
            return Err(format!(
                "baked model: truncated at byte {} (wanted {n} more, {} left)",
                self.at,
                self.b.len().saturating_sub(self.at)
            ));
        }
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn usize(&mut self) -> Result<usize, String> {
        Ok(self.u32()? as usize)
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn bool(&mut self) -> Result<bool, String> {
        Ok(self.take(1)?[0] != 0)
    }
    fn str(&mut self) -> Result<String, String> {
        let n = self.usize()?;
        let s = self.take(n)?;
        String::from_utf8(s.to_vec()).map_err(|e| format!("baked model: bad text: {e}"))
    }
    fn pod<T: bytemuck::Pod>(&mut self) -> Result<Vec<T>, String> {
        let n = self.usize()?;
        let bytes = self.take(n * std::mem::size_of::<T>())?;
        // `pod_collect_to_vec` rather than a cast: the buffer is not guaranteed
        // to be aligned for T, and a cast would be a hard error on the day a
        // caller reads out of a mapped file.
        Ok(bytemuck::pod_collect_to_vec(bytes))
    }
    fn f32s(&mut self) -> Result<Vec<f32>, String> {
        self.pod()
    }
    fn arr<const N: usize>(&mut self) -> Result<[f32; N], String> {
        let mut a = [0.0; N];
        for v in a.iter_mut() {
            *v = self.f32()?;
        }
        Ok(a)
    }
    fn opt_tex(&mut self) -> Result<Option<crate::model::Tex>, String> {
        if !self.bool()? {
            return Ok(None);
        }
        let png: Vec<u8> = self.pod()?;
        crate::model::decode_texture(&png)
            .map(Some)
            .ok_or_else(|| "baked model: an embedded texture will not decode".to_string())
    }
    fn interp(&mut self) -> Result<Interp, String> {
        match self.u32()? {
            0 => Ok(Interp::Linear),
            1 => Ok(Interp::Step),
            n => Err(format!("baked model: unknown interpolation {n}")),
        }
    }
}

/// Decode a baked file.
pub fn read(bytes: &[u8]) -> Result<Model, String> {
    let mut r = R { b: bytes, at: 0 };
    let magic = r.u32()?;
    if magic != MAGIC {
        return Err("baked model: not an Aurora model file".into());
    }
    let version = r.u32()?;
    if version != VERSION {
        return Err(format!(
            "baked model: version {version}, this build reads {VERSION} - bake it again"
        ));
    }

    let n = r.usize()?;
    let mut primitives = Vec::with_capacity(n);
    for _ in 0..n {
        let vertices: Vec<Vertex> = r.pod()?;
        let indices: Vec<u32> = r.pod()?;
        let material = r.str()?;
        let base_color = r.arr::<4>()?;
        let metallic = r.f32()?;
        let roughness = r.f32()?;
        let emissive = r.arr::<3>()?;
        let texture = r.opt_tex()?;
        let normal_tex = r.opt_tex()?;
        let mr_tex = r.opt_tex()?;
        let emissive_tex = r.opt_tex()?;
        let skinned = r.bool()?;
        primitives.push(Primitive {
            mesh: MeshData { vertices, indices },
            material,
            base_color,
            metallic,
            roughness,
            emissive,
            texture,
            normal_tex,
            mr_tex,
            emissive_tex,
            skinned,
        });
    }

    let skeleton = if r.bool()? {
        let n = r.usize()?;
        let mut joints = Vec::with_capacity(n);
        for _ in 0..n {
            let p = r.u32()?;
            let parent = if p == u32::MAX {
                None
            } else {
                Some(p as usize)
            };
            let ib = r.arr::<16>()?;
            let t = r.arr::<3>()?;
            let rot = r.arr::<4>()?;
            let s = r.arr::<3>()?;
            let name = r.str()?;
            joints.push(Joint {
                parent,
                inverse_bind: Mat4::from_cols_array(&ib),
                t: Vec3::from_array(t),
                r: Quat::from_array(rot),
                s: Vec3::from_array(s),
                name,
            });
        }
        Some(Skeleton { joints })
    } else {
        None
    };

    let n = r.usize()?;
    let mut clips = Vec::with_capacity(n);
    for _ in 0..n {
        let name = r.str()?;
        let duration = r.f32()?;
        let nc = r.usize()?;
        let mut channels = Vec::with_capacity(nc);
        for _ in 0..nc {
            let joint = r.usize()?;
            let path = match r.u32()? {
                0 => Path::Translation,
                1 => Path::Rotation,
                2 => Path::Scale,
                n => return Err(format!("baked model: unknown channel path {n}")),
            };
            let interp = r.interp()?;
            let times = r.f32s()?;
            let values = r.f32s()?;
            channels.push(Channel {
                joint,
                path,
                interp,
                times,
                values,
            });
        }
        let root = if r.bool()? {
            let interp = r.interp()?;
            let times = r.f32s()?;
            let values = r.f32s()?;
            Some(RootMotion {
                interp,
                times,
                values,
            })
        } else {
            None
        };
        clips.push(Clip {
            name,
            duration,
            channels,
            root,
        });
    }

    if r.at != bytes.len() {
        return Err(format!(
            "baked model: {} bytes left over - the file is longer than its contents",
            bytes.len() - r.at
        ));
    }
    Ok(Model {
        primitives,
        skeleton,
        clips,
    })
}

/// Where the bake for `src` lives: the same path with the extension replaced.
///
/// Beside the source rather than in a build directory, deliberately. A bake is
/// derived and disposable, and the one thing that must never happen is a bake
/// that has drifted from the art it came from being picked up silently - so it
/// sits where anyone looking at the art will see it, and [`newer_than_source`]
/// decides whether it may be used.
pub fn baked_path(src: &str) -> std::path::PathBuf {
    std::path::Path::new(src).with_extension(EXT)
}

/// May this bake be used for that source?
///
/// Only if it exists and is no older than the source file. Re-baking is cheap
/// and being wrong is not: an edited model that keeps loading its stale bake is
/// a change that does not take, and the hunt for that starts everywhere except
/// the file that was never re-read.
///
/// A source that cannot be stat-ed at all - the art is not installed, the game
/// ships baked only - counts as unchanged, because then the bake is all there is.
pub fn usable(src: &str, baked: &std::path::Path) -> bool {
    let Ok(b) = std::fs::metadata(baked) else {
        return false;
    };
    let Ok(bt) = b.modified() else {
        return false;
    };
    match std::fs::metadata(src).and_then(|m| m.modified()) {
        Ok(st) => bt >= st,
        Err(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Model {
        let mut v = Vertex::new([1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [0.25, 0.5]);
        v.joints = [3, 1, 4, 1];
        v.weights = [0.5, 0.25, 0.15, 0.1];
        v.tangent = [1.0, 0.0, 0.0, -1.0];
        Model {
            primitives: vec![Primitive {
                mesh: MeshData {
                    vertices: vec![v, v, v],
                    indices: vec![0, 1, 2],
                },
                material: "Wall71".into(),
                base_color: [0.1, 0.2, 0.3, 1.0],
                metallic: 0.5,
                roughness: 0.75,
                emissive: [0.0, 0.5, 1.0],
                texture: Some((vec![1, 2, 3, 4, 5, 6, 7, 8], 2, 1)),
                normal_tex: None,
                mr_tex: None,
                emissive_tex: None,
                skinned: true,
            }],
            skeleton: Some(Skeleton {
                joints: vec![
                    Joint {
                        parent: None,
                        inverse_bind: Mat4::IDENTITY,
                        t: Vec3::new(0.0, 1.0, 0.0),
                        r: Quat::IDENTITY,
                        s: Vec3::ONE,
                        name: "Pelvis".into(),
                    },
                    Joint {
                        parent: Some(0),
                        inverse_bind: Mat4::from_translation(Vec3::X),
                        t: Vec3::new(0.0, 0.5, 0.0),
                        r: Quat::from_rotation_y(0.5),
                        s: Vec3::splat(2.0),
                        name: "Spine_01".into(),
                    },
                ],
            }),
            clips: vec![Clip {
                name: "A_Walk_F_Masc".into(),
                duration: 1.25,
                channels: vec![Channel {
                    joint: 1,
                    path: Path::Rotation,
                    interp: Interp::Step,
                    times: vec![0.0, 0.5, 1.0],
                    values: vec![0.0; 12],
                }],
                root: Some(RootMotion {
                    interp: Interp::Linear,
                    times: vec![0.0, 1.25],
                    values: vec![0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
                }),
            }],
        }
    }

    // Everything that goes in comes back. Field by field, because the failure
    // this guards is one field of one joint quietly reading as another - which
    // draws a model that is almost right, and there is no worse kind.
    #[test]
    fn a_model_survives_the_round_trip() {
        let m = sample();
        let back = read(&write(&m)).expect("round trip");

        assert_eq!(back.primitives.len(), 1);
        let (a, b) = (&m.primitives[0], &back.primitives[0]);
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&a.mesh.vertices),
            bytemuck::cast_slice::<_, u8>(&b.mesh.vertices),
            "vertices are the exact bytes the GPU is given"
        );
        assert_eq!(a.mesh.indices, b.mesh.indices);
        assert_eq!(a.material, b.material);
        assert_eq!(a.base_color, b.base_color);
        assert_eq!(a.metallic, b.metallic);
        assert_eq!(a.roughness, b.roughness);
        assert_eq!(a.emissive, b.emissive);
        assert_eq!(a.texture, b.texture);
        assert!(b.normal_tex.is_none() && b.mr_tex.is_none() && b.emissive_tex.is_none());
        assert!(b.skinned);

        let (sa, sb) = (m.skeleton.unwrap(), back.skeleton.unwrap());
        assert_eq!(sa.joints.len(), sb.joints.len());
        for (ja, jb) in sa.joints.iter().zip(sb.joints.iter()) {
            // The root's `None` must not come back as joint 0, which is a real
            // joint and would silently reparent the whole rig onto itself.
            assert_eq!(ja.parent, jb.parent);
            assert_eq!(ja.inverse_bind, jb.inverse_bind);
            assert_eq!(ja.t, jb.t);
            assert_eq!(ja.r, jb.r);
            assert_eq!(ja.s, jb.s);
            assert_eq!(ja.name, jb.name);
        }

        assert_eq!(back.clips.len(), 1);
        let (ca, cb) = (&m.clips[0], &back.clips[0]);
        assert_eq!(ca.name, cb.name);
        assert_eq!(ca.duration, cb.duration);
        assert_eq!(ca.channels.len(), cb.channels.len());
        let (cha, chb) = (&ca.channels[0], &cb.channels[0]);
        assert_eq!(cha.joint, chb.joint);
        assert_eq!(cha.path, chb.path);
        assert_eq!(cha.interp, chb.interp);
        assert_eq!(cha.times, chb.times);
        assert_eq!(cha.values, chb.values);
        let (ra, rb) = (ca.root.as_ref().unwrap(), cb.root.as_ref().unwrap());
        assert_eq!(ra.interp, rb.interp);
        assert_eq!(ra.times, rb.times);
        assert_eq!(ra.values, rb.values);
    }

    // A model with nothing in it is a real case - a mesh-only file has no
    // skeleton and no clips - and an encoding that only works when every
    // optional part is present is an encoding that fails on the common file.
    #[test]
    fn an_empty_model_round_trips_too() {
        let m = Model {
            primitives: Vec::new(),
            skeleton: None,
            clips: Vec::new(),
        };
        let back = read(&write(&m)).expect("round trip");
        assert!(back.primitives.is_empty());
        assert!(back.skeleton.is_none());
        assert!(back.clips.is_empty());
    }

    // NOT AN AURORA FILE, a version from another build, and a half-written one.
    //
    // All three are the realistic failures for a generated file, and all three
    // have to be errors rather than a model made of whatever the bytes meant.
    #[test]
    fn a_bad_bake_is_refused_rather_than_reinterpreted() {
        assert!(read(b"not a model at all").is_err(), "wrong magic");

        let mut wrong = write(&sample());
        wrong[4..8].copy_from_slice(&(VERSION + 1).to_le_bytes());
        let e = match read(&wrong) {
            Err(e) => e,
            Ok(_) => panic!("a future version must not be read as this one"),
        };
        assert!(e.contains("bake it again"), "says what to do about it: {e}");

        let good = write(&sample());
        for cut in [8, good.len() / 2, good.len() - 1] {
            assert!(
                read(&good[..cut]).is_err(),
                "a file truncated at {cut} must not read as a model"
            );
        }

        let mut extra = write(&sample());
        extra.push(0);
        assert!(
            read(&extra).is_err(),
            "trailing bytes mean this is not the file it claims to be"
        );
    }
}
