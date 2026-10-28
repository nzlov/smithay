use std::collections::VecDeque;

use super::{BufferCoords, BufferTrait, Dmabuf, ErasedContextId, Fourcc, Size};

// Two recent sizes let different-sized outputs share a GPU pair without repeatedly reallocating,
// while retaining at most one more intermediate buffer than the single-buffer cache.
const MAX_CACHED_BUFFERS: usize = 2;

#[derive(Debug, Default)]
pub(super) struct DmabufCache {
    contexts: Option<(ErasedContextId, ErasedContextId)>,
    buffers: VecDeque<(bool, Dmabuf)>,
}

impl DmabufCache {
    pub(super) fn set_contexts(&mut self, render: ErasedContextId, target: ErasedContextId) {
        let contexts = (render, target);
        if self.contexts.as_ref() != Some(&contexts) {
            self.clear();
            self.contexts = Some(contexts);
        }
    }

    pub(super) fn clear(&mut self) {
        self.buffers.clear();
        self.contexts = None;
    }

    pub(super) fn get_or_insert_with<E>(
        &mut self,
        size: Size<i32, BufferCoords>,
        format: Fourcc,
        create: impl FnOnce() -> Result<(bool, Dmabuf), E>,
    ) -> Result<&mut (bool, Dmabuf), E> {
        if let Some(index) = self
            .buffers
            .iter()
            .position(|(_, buffer)| buffer.size() == size && buffer.format().code == format)
        {
            let buffer = self.buffers.remove(index).unwrap();
            self.buffers.push_front(buffer);
        } else {
            // Preserve the existing entries if allocation fails.
            let buffer = create()?;
            if self.buffers.len() == MAX_CACHED_BUFFERS {
                self.buffers.pop_back();
            }
            self.buffers.push_front(buffer);
        }

        Ok(self.buffers.front_mut().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::convert::Infallible;

    use super::*;
    use crate::backend::allocator::{Modifier, dmabuf::DmabufFlags};
    use crate::backend::renderer::{ContextId, test::DummyTexture};

    const UHD: (i32, i32) = (3840, 2160);
    const QHD: (i32, i32) = (2560, 1440);
    const FHD: (i32, i32) = (1920, 1080);
    const SDR: Fourcc = Fourcc::Abgr8888;
    const HDR: Fourcc = Fourcc::Abgr2101010;

    fn copy_buffer(size: Size<i32, BufferCoords>, format: Fourcc) -> Dmabuf {
        // The cache only needs buffer identity and metadata, so a sparse memfd needs no GPU.
        let fd = rustix::fs::memfd_create("copy-buffer-cache", rustix::fs::MemfdFlags::CLOEXEC).unwrap();
        let stride = size.w as u32 * 4;
        rustix::fs::ftruncate(&fd, u64::from(stride) * size.h as u64).unwrap();
        let mut builder = Dmabuf::builder(size, format, Modifier::Linear, DmabufFlags::empty());
        assert!(builder.add_plane(fd, 0, stride));
        builder.build().unwrap()
    }

    fn get_buffer(
        cache: &mut DmabufCache,
        allocations: &Cell<usize>,
        size: (i32, i32),
        format: Fourcc,
        direct: bool,
    ) -> (bool, Dmabuf) {
        let size = size.into();
        let (direct, buffer) = cache
            .get_or_insert_with(size, format, || {
                allocations.set(allocations.get() + 1);
                Ok::<_, Infallible>((direct, copy_buffer(size, format)))
            })
            .unwrap();
        (*direct, buffer.clone())
    }

    #[test]
    fn alternating_output_sizes_reuse_copy_buffers() {
        let mut cache = DmabufCache::default();
        let allocations = Cell::new(0);
        let (_, uhd) = get_buffer(&mut cache, &allocations, UHD, SDR, true);
        let (_, qhd) = get_buffer(&mut cache, &allocations, QHD, SDR, true);

        for _ in 0..10 {
            assert_eq!(get_buffer(&mut cache, &allocations, UHD, SDR, true).1, uhd);
            assert_eq!(get_buffer(&mut cache, &allocations, QHD, SDR, true).1, qhd);
        }
        assert_eq!(allocations.get(), 2);
    }

    #[test]
    fn incompatible_formats_use_separate_copy_buffers() {
        let mut cache = DmabufCache::default();
        let allocations = Cell::new(0);
        let (_, sdr) = get_buffer(&mut cache, &allocations, UHD, SDR, true);
        let (_, hdr) = get_buffer(&mut cache, &allocations, UHD, HDR, true);

        assert_ne!(sdr, hdr);
        assert_eq!(hdr.format().code, HDR);
        assert_eq!(get_buffer(&mut cache, &allocations, UHD, SDR, true).1, sdr);
        assert_eq!(allocations.get(), 2);
    }

    #[test]
    fn cache_hits_keep_recent_output_sizes() {
        let mut cache = DmabufCache::default();
        let allocations = Cell::new(0);
        let (_, uhd) = get_buffer(&mut cache, &allocations, UHD, SDR, true);
        let qhd = get_buffer(&mut cache, &allocations, QHD, SDR, true).1.weak();

        assert_eq!(get_buffer(&mut cache, &allocations, UHD, SDR, true).1, uhd);
        get_buffer(&mut cache, &allocations, FHD, SDR, true);
        assert!(qhd.upgrade().is_none());
        assert_eq!(get_buffer(&mut cache, &allocations, UHD, SDR, true).1, uhd);
        assert_eq!(allocations.get(), 3);

        get_buffer(&mut cache, &allocations, QHD, SDR, true);
        assert_eq!(allocations.get(), 4);
    }

    #[test]
    fn clearing_cache_releases_copy_buffers() {
        let mut cache = DmabufCache::default();
        let allocations = Cell::new(0);
        let buffer = get_buffer(&mut cache, &allocations, UHD, SDR, true).1.weak();

        cache.clear();
        assert!(buffer.upgrade().is_none());
        get_buffer(&mut cache, &allocations, UHD, SDR, true);
        assert_eq!(allocations.get(), 2);
    }

    #[test]
    fn dma_and_cpu_copy_choices_survive_cache_hits() {
        let mut cache = DmabufCache::default();
        let allocations = Cell::new(0);
        get_buffer(&mut cache, &allocations, UHD, SDR, true);
        get_buffer(&mut cache, &allocations, QHD, SDR, false);

        assert!(get_buffer(&mut cache, &allocations, UHD, SDR, false).0);
        assert!(!get_buffer(&mut cache, &allocations, QHD, SDR, true).0);
        assert_eq!(allocations.get(), 2);
    }

    #[test]
    fn failed_allocation_preserves_cached_copy_buffers() {
        let mut cache = DmabufCache::default();
        let allocations = Cell::new(0);
        let (_, uhd) = get_buffer(&mut cache, &allocations, UHD, SDR, true);
        let (_, qhd) = get_buffer(&mut cache, &allocations, QHD, SDR, true);

        assert!(
            cache
                .get_or_insert_with(FHD.into(), SDR, || Err("allocation failed"))
                .is_err()
        );
        assert_eq!(get_buffer(&mut cache, &allocations, UHD, SDR, true).1, uhd);
        assert_eq!(get_buffer(&mut cache, &allocations, QHD, SDR, true).1, qhd);
        assert_eq!(allocations.get(), 2);
    }

    #[test]
    fn renderer_context_changes_discard_copy_buffers() {
        let mut cache = DmabufCache::default();
        let allocations = Cell::new(0);
        let render = ContextId::<DummyTexture>::new().erased();
        let target = ContextId::<DummyTexture>::new().erased();
        cache.set_contexts(render.clone(), target.clone());
        let (_, buffer) = get_buffer(&mut cache, &allocations, UHD, SDR, true);
        cache.set_contexts(render.clone(), target);
        assert_eq!(get_buffer(&mut cache, &allocations, UHD, SDR, true).1, buffer);

        let target = ContextId::<DummyTexture>::new().erased();
        cache.set_contexts(render, target.clone());
        assert_ne!(get_buffer(&mut cache, &allocations, UHD, SDR, true).1, buffer);
        cache.set_contexts(ContextId::<DummyTexture>::new().erased(), target);
        get_buffer(&mut cache, &allocations, UHD, SDR, true);
        assert_eq!(allocations.get(), 3);
    }
}
