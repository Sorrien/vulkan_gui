use ash::vk;
use std::sync::Arc;

use crate::ash_bootstrap::LogicalDevice;

pub struct Descriptor {
    pub set: vk::DescriptorSet,
    pub layout: DescriptorLayout,
}

impl Descriptor {
    pub fn new(
        device: Arc<LogicalDevice>,
        set: vk::DescriptorSet,
        layout: vk::DescriptorSetLayout,
    ) -> Self {
        Self {
            set,
            layout: DescriptorLayout::new(device, layout),
        }
    }
}

pub struct DescriptorLayout {
    device: Arc<LogicalDevice>,
    pub handle: vk::DescriptorSetLayout,
}

impl DescriptorLayout {
    pub fn new(device: Arc<LogicalDevice>, layout: vk::DescriptorSetLayout) -> Self {
        Self {
            device,
            handle: layout,
        }
    }
}

impl Drop for DescriptorLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .handle
                .destroy_descriptor_set_layout(self.handle, None)
        };
    }
}

pub struct DescriptorLayoutBuilder<'a> {
    bindings: Vec<vk::DescriptorSetLayoutBinding<'a>>,
}
impl DescriptorLayoutBuilder<'_> {
    pub fn new() -> Self {
        Self { bindings: vec![] }
    }

    pub fn add_binding(mut self, binding: u32, descriptor_type: vk::DescriptorType) -> Self {
        let new_bind = vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_count(1)
            .descriptor_type(descriptor_type);
        self.bindings.push(new_bind);
        self
    }

    pub fn clear(&mut self) {
        self.bindings.clear();
    }

    pub fn build(
        mut self,
        device: Arc<LogicalDevice>,
        shader_stages: vk::ShaderStageFlags,
    ) -> Result<vk::DescriptorSetLayout, vk::Result> {
        self.bindings.iter_mut().for_each(|binding| {
            binding.stage_flags |= shader_stages;
        });

        let info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&self.bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::empty());

        unsafe { device.handle.create_descriptor_set_layout(&info, None) }
    }
}

pub struct DescriptorAllocator {
    device: Arc<LogicalDevice>,
    pool: Option<vk::DescriptorPool>,
}

impl DescriptorAllocator {
    pub fn new(device: Arc<LogicalDevice>) -> Self {
        Self { device, pool: None }
    }

    pub fn init_pool(&mut self, max_sets: u32, pool_ratios: Vec<PoolSizeRatio>) {
        let pool_sizes = pool_ratios
            .iter()
            .map(|pool_ratio| {
                vk::DescriptorPoolSize::default()
                    .ty(pool_ratio.desc_type)
                    .descriptor_count((pool_ratio.ratio * (max_sets as f32)) as u32)
            })
            .collect::<Vec<_>>();

        let create_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::empty())
            .max_sets(max_sets)
            .pool_sizes(&pool_sizes);

        self.pool = Some(
            unsafe {
                self.device
                    .handle
                    .create_descriptor_pool(&create_info, None)
            }
            .expect("failed to create descriptor pool!"),
        );
    }

    pub fn clear_descriptors(&mut self) {
        if let Some(pool) = self.pool {
            unsafe {
                self.device
                    .handle
                    .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())
            }
            .expect("failed to reset descriptor pool!");
        }
    }

    pub fn destroy_pool(&mut self) {
        if let Some(pool) = self.pool {
            unsafe { self.device.handle.destroy_descriptor_pool(pool, None) };
        }
    }

    pub fn allocate(
        &self,
        layouts: Vec<vk::DescriptorSetLayout>,
    ) -> Option<Vec<vk::DescriptorSet>> {
        if let Some(pool) = self.pool {
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);

            Some(
                unsafe { self.device.handle.allocate_descriptor_sets(&alloc_info) }
                    .expect("failed to allocate descriptor sets!"),
            )
        } else {
            None
        }
    }
}

impl Drop for DescriptorAllocator {
    fn drop(&mut self) {
        if let Some(pool) = self.pool {
            unsafe { self.device.handle.destroy_descriptor_pool(pool, None) }
        }
    }
}

#[derive(Clone)]
pub struct PoolSizeRatio {
    pub desc_type: vk::DescriptorType,
    pub ratio: f32,
}

impl PoolSizeRatio {
    pub fn new(desc_type: vk::DescriptorType, ratio: f32) -> Self {
        Self { desc_type, ratio }
    }
}

#[derive(Clone)]
pub struct DescriptorAllocatorGrowable {
    ratios: Vec<PoolSizeRatio>,
    full_pools: Vec<vk::DescriptorPool>,
    ready_pools: Vec<vk::DescriptorPool>,
    sets_per_pool: u32,
    device: Arc<LogicalDevice>,
}

impl DescriptorAllocatorGrowable {
    pub fn new(device: Arc<LogicalDevice>) -> Self {
        Self {
            device,
            ratios: vec![],
            full_pools: vec![],
            ready_pools: vec![],
            sets_per_pool: 0,
        }
    }

    pub fn get_pool(&mut self) -> vk::DescriptorPool {
        let new_pool = if let Some(pool) = self.ready_pools.pop() {
            pool
        } else {
            let pool = self
                .create_pool(self.sets_per_pool, &self.ratios)
                .expect("failed to create new descriptor pool!");

            self.sets_per_pool = (self.sets_per_pool as f32 * 1.5) as u32;

            if self.sets_per_pool > 4092 {
                self.sets_per_pool = 4092;
            }
            pool
        };
        new_pool
    }

    pub fn create_pool(
        &self,
        set_count: u32,
        pool_ratios: &Vec<PoolSizeRatio>,
    ) -> Result<vk::DescriptorPool, vk::Result> {
        let pool_sizes = pool_ratios
            .iter()
            .map(|ratio| {
                vk::DescriptorPoolSize::default()
                    .ty(ratio.desc_type)
                    .descriptor_count((ratio.ratio * set_count as f32) as u32)
            })
            .collect::<Vec<_>>();

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::empty())
            .max_sets(set_count)
            .pool_sizes(&pool_sizes);

        let new_pool_result =
            unsafe { self.device.handle.create_descriptor_pool(&pool_info, None) };
        new_pool_result
    }

    pub fn init(&mut self, max_sets: u32, pool_ratios: Vec<PoolSizeRatio>) {
        self.ratios = pool_ratios;

        let new_pool = self
            .create_pool(max_sets, &self.ratios)
            .expect("failed to create new descriptor pool!");

        self.sets_per_pool = ((max_sets as f32) * 1.5) as u32;
        self.ready_pools.push(new_pool);
    }

    pub fn clear_pools(&mut self) {
        reset_descriptor_pools(self.device.clone(), &self.ready_pools);
        reset_descriptor_pools(self.device.clone(), &self.full_pools);

        self.ready_pools.append(&mut self.full_pools);
    }

    pub fn allocate(&mut self, layout: vk::DescriptorSetLayout) -> vk::DescriptorSet {
        let pool = self.get_pool();

        let layouts = [layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);

        let result = unsafe { self.device.handle.allocate_descriptor_sets(&alloc_info) };

        let ds = match result {
            Ok(ds) => {
                self.ready_pools.push(pool);
                ds
            }
            Err(result) => {
                if result == vk::Result::ERROR_OUT_OF_POOL_MEMORY_KHR
                    || result == vk::Result::ERROR_FRAGMENTED_POOL
                {
                    self.full_pools.push(pool);

                    let new_pool = self.get_pool();
                    let alloc_info = alloc_info.descriptor_pool(new_pool);

                    unsafe { self.device.handle.allocate_descriptor_sets(&alloc_info) }
                        .expect("failed to allocate descriptor sets")
                } else {
                    panic!("failed to allocate descriptor sets with unhandled error!");
                }
            }
        };

        ds[0]
    }

    pub fn destroy_pools(&mut self) {
        destroy_descriptor_pools(self.device.clone(), &mut self.ready_pools);
        destroy_descriptor_pools(self.device.clone(), &mut self.full_pools);
    }
}

pub fn reset_descriptor_pools(device: Arc<LogicalDevice>, pools: &Vec<vk::DescriptorPool>) {
    for p in pools {
        unsafe {
            device
                .handle
                .reset_descriptor_pool(*p, vk::DescriptorPoolResetFlags::empty())
                .expect("failed to reset descriptor pool!")
        }
    }
}

pub fn destroy_descriptor_pools(device: Arc<LogicalDevice>, pools: &mut Vec<vk::DescriptorPool>) {
    for pool in pools.drain(..) {
        unsafe { device.handle.destroy_descriptor_pool(pool, None) }
    }
}

impl Drop for DescriptorAllocatorGrowable {
    fn drop(&mut self) {
        self.destroy_pools();
    }
}

pub struct DescriptorWriter<'a> {
    buffer_infos: Vec<[vk::DescriptorBufferInfo; 1]>,
    image_infos: Vec<[vk::DescriptorImageInfo; 1]>,
    writes: Vec<(vk::WriteDescriptorSet<'a>, WriteDescriptorInfo)>,
}

impl DescriptorWriter<'_> {
    pub fn new() -> Self {
        Self {
            buffer_infos: Vec::new(),
            image_infos: Vec::new(),
            writes: Vec::new(),
        }
    }

    pub fn write_buffer(
        &mut self,
        binding: u32,
        buffer: vk::Buffer,
        size: vk::DeviceSize,
        offset: u64,
        desc_type: BufferDescriptorType,
    ) {
        let info = vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(offset)
            .range(size);
        let infos: [vk::DescriptorBufferInfo; 1] = [info];
        self.buffer_infos.push(infos);

        let write: vk::WriteDescriptorSet<'_> = vk::WriteDescriptorSet::default()
            .dst_binding(binding)
            .dst_set(vk::DescriptorSet::null())
            .descriptor_count(1)
            .descriptor_type(desc_type.into());

        self.writes.push((
            write,
            WriteDescriptorInfo::Buffer(self.buffer_infos.len() - 1),
        ));
    }

    pub fn write_image(
        &mut self,
        binding: u32,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
        image_layout: vk::ImageLayout,
        desc_type: vk::DescriptorType,
    ) {
        let info = vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(image_view)
            .image_layout(image_layout);
        let infos: [vk::DescriptorImageInfo; 1] = [info];

        let write: vk::WriteDescriptorSet<'_> = vk::WriteDescriptorSet::default()
            .dst_binding(binding)
            .dst_set(vk::DescriptorSet::null())
            .descriptor_count(1)
            .descriptor_type(desc_type);

        self.image_infos.push(infos);

        self.writes.push((
            write,
            WriteDescriptorInfo::Image(self.image_infos.len() - 1),
        ));
    }

    pub fn update_set(&mut self, device: Arc<LogicalDevice>, set: vk::DescriptorSet) {
        /*         self.writes.iter_mut().for_each(|(mut write, write_info)| {
            write = match write_info {
                WriteDescriptorInfo::Image(i) => {
                    write.image_info(self.image_infos.get(*i).unwrap())
                }
                WriteDescriptorInfo::Buffer(i) => {
                    write.buffer_info(self.buffer_infos.get(*i).unwrap())
                }
            }
            .dst_set(set);
        });

        let writes = self
            .writes
            .iter()
            .map(|(write, _)| *write)
            .collect::<Vec<_>>(); */
        let writes = self
            .writes
            .iter_mut()
            .map(|(write, write_info)| {
                match write_info {
                    WriteDescriptorInfo::Image(i) => {
                        write.image_info(self.image_infos.get(*i).unwrap())
                    }
                    WriteDescriptorInfo::Buffer(i) => {
                        write.buffer_info(self.buffer_infos.get(*i).unwrap())
                    }
                }
                .dst_set(set)
            })
            .collect::<Vec<_>>();
        unsafe { device.handle.update_descriptor_sets(&writes, &[]) };
    }
}

enum WriteDescriptorInfo {
    Image(usize),
    Buffer(usize),
}

pub enum BufferDescriptorType {
    UniformBuffer,
    StorageBuffer,
    UniformBufferDynamic,
    StorageBufferDynamic,
}

impl Into<vk::DescriptorType> for BufferDescriptorType {
    fn into(self) -> vk::DescriptorType {
        match self {
            BufferDescriptorType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
            BufferDescriptorType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
            BufferDescriptorType::UniformBufferDynamic => {
                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
            }
            BufferDescriptorType::StorageBufferDynamic => {
                vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
            }
        }
    }
}
