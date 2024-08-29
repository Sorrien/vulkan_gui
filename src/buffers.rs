use std::{
    mem::{self, ManuallyDrop},
    ptr::{self, NonNull},
    sync::{Arc, Mutex},
};

use ash::{util::Align, vk};
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
    MemoryLocation,
};

use crate::ash_bootstrap::LogicalDevice;

pub struct AllocatedBuffer {
    device: Arc<LogicalDevice>,
    allocator: Arc<Mutex<Allocator>>,
    pub buffer: vk::Buffer,
    pub allocation: ManuallyDrop<Allocation>,
}

impl AllocatedBuffer {
    pub fn new(
        device: Arc<LogicalDevice>,
        allocator: Arc<Mutex<Allocator>>,
        name: &str,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Self {
        let buffer_info = vk::BufferCreateInfo::default().size(size).usage(usage);
        let buffer = unsafe { device.handle.create_buffer(&buffer_info, None) }
            .expect("failed to create buffer!");

        let requirements = unsafe { device.handle.get_buffer_memory_requirements(buffer) };

        let allocation = allocator
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: true, // Buffers are always linear
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .expect("failed to allocate buffer!");

        unsafe {
            device
                .handle
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }
        .expect("failed to bind buffer!");

        Self {
            device,
            allocator,
            buffer,
            allocation: ManuallyDrop::new(allocation),
        }
    }

    pub fn mapped_ptr<T: Sized>(&self) -> Option<NonNull<T>> {
        if let Some(mapped_ptr) = self.allocation.mapped_ptr() {
            Some(mapped_ptr.cast::<T>())
        } else {
            None
        }
    }
}

impl Drop for AllocatedBuffer {
    fn drop(&mut self) {
        let allocation = unsafe { ManuallyDrop::take(&mut self.allocation) };
        self.allocator
            .lock()
            .unwrap()
            .free(allocation)
            .expect("failed to free memory for allocated image!");
        unsafe {
            self.device.handle.destroy_buffer(self.buffer, None);
        }
    }
}

pub fn copy_to_cpu_buffer<DataType: std::marker::Copy>(
    staging_buffer: &AllocatedBuffer,
    size: vk::DeviceSize,
    data: &[DataType],
) {
    let ptr = unsafe { staging_buffer.allocation.mapped_ptr().unwrap().as_mut() };
    let mut align =
        unsafe { Align::new(ptr, std::mem::align_of::<DataType>() as u64, size as u64) };
    align.copy_from_slice(&data);
}

pub fn write_to_cpu_buffer<T>(data: &T, allocated_buffer: &mut AllocatedBuffer) {
    let dst_ptr = allocated_buffer.allocation.mapped_ptr().unwrap().as_ptr();
    let size = mem::size_of::<T>();
    unsafe { ptr::copy_nonoverlapping(data as *const _ as *const u8, dst_ptr as *mut u8, size) };
}

pub fn copy_buffer_to_image(
    command_buffer: vk::CommandBuffer,
    device: Arc<LogicalDevice>,
    buffer: vk::Buffer,
    image: vk::Image,
    image_extent: vk::Extent3D,
) {
    let buffer_image_copy = create_buffer_copy_region(image_extent, 0, 0, 0);

    let regions = [buffer_image_copy];
    copy_buffer_regions_to_image(command_buffer, device, buffer, image, &regions);
}

pub fn copy_buffer_regions_to_image(
    command_buffer: vk::CommandBuffer,
    device: Arc<LogicalDevice>,
    buffer: vk::Buffer,
    image: vk::Image,
    regions: &[vk::BufferImageCopy],
) {
    unsafe {
        device.handle.cmd_copy_buffer_to_image(
            command_buffer,
            buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        )
    }
}

pub fn create_buffer_copy_region(
    image_extent: vk::Extent3D,
    offset: vk::DeviceSize,
    mip_level: u32,
    base_array_layer: u32,
) -> vk::BufferImageCopy {
    let buffer_image_copy = vk::BufferImageCopy::default()
        .buffer_offset(offset)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(mip_level)
                .base_array_layer(base_array_layer)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D::default())
        .image_extent(image_extent.into());

    buffer_image_copy
}

// Simple offset_of macro akin to C++ offsetof
#[macro_export]
macro_rules! offset_of {
    ($base:path, $field:ident) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let b: $base = std::mem::zeroed();
            std::ptr::addr_of!(b.$field) as isize - std::ptr::addr_of!(b) as isize
        }
    }};
}
pub use offset_of;
