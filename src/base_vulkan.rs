use std::{
    ffi::{CStr, CString},
    fs::File,
    mem::{size_of, ManuallyDrop},
    path::Path,
    sync::{Arc, Mutex},
};

use ash::{
    //extensions::khr::Swapchain,
    khr::swapchain::Device as Swapchain,
    util::{read_spv, Align},
    vk, Entry,
};
use ash_bootstrap::{
    Instance, InstanceBuilder, LogicalDevice, PhysicalDeviceSelector, QueueFamilyIndices,
    VulkanSurface,
};
use debug::DebugMessenger;
use gpu_allocator::{vulkan::*, MemoryLocation};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use swapchain::{MySwapchain, SwapchainBuilder, SwapchainSupportDetails};
use winit::window::Window;

use crate::{
    ash_bootstrap,
    buffers::{copy_to_cpu_buffer, AllocatedBuffer, GPUDrawPushConstants, GPUMeshBuffers, Vertex},
    debug,
    descriptors::{
        Descriptor, DescriptorAllocator, DescriptorAllocatorGrowable, DescriptorLayout,
        DescriptorLayoutBuilder, DescriptorWriter, PoolSizeRatio,
    },
    pipelines::{Pipeline, PipelineBuilder, PipelineLayout},
    swapchain,
    AllocatedImage, ComputeEffect, ComputePushConstants, GPUSceneData,
};

#[derive(Debug)]
pub enum ShaderModuleError {
    Unknown,
    FileError(std::io::Error),
    ReadSPVError(std::io::Error),
    CreateShaderModuleError(ash::vk::Result),
}

pub struct BaseVulkanState {
    pub msaa_samples: vk::SampleCountFlags,
    pub queue_family_indices: QueueFamilyIndices,
    pub swapchain_support: SwapchainSupportDetails,
    pub allocator: Arc<Mutex<Allocator>>,
    pub graphics_queue: vk::Queue,
    pub device: Arc<LogicalDevice>,
    pub physical_device: vk::PhysicalDevice,
    pub debug_messenger: DebugMessenger,
    pub surface: Arc<VulkanSurface>,
    pub instance: Arc<Instance>,
    pub entry: Entry,
    pub window: Window,
}

impl BaseVulkanState {
    pub fn new(window: Window, application_title: String) -> Self {
        #[cfg(feature = "validation_layers")]
        let enable_validation_layers = true;
        #[cfg(not(feature = "validation_layers"))]
        let enable_validation_layers = false;

        let entry = unsafe { ash::Entry::load() }.expect("vulkan entry failed to load!");
        let instance = InstanceBuilder::new()
            .entry(entry.clone())
            .application_name(application_title)
            .api_version(vk::API_VERSION_1_3)
            .raw_display_handle(window.raw_display_handle().unwrap())
            .enable_validation_layers(enable_validation_layers)
            .build();

        let debug_messenger =
            DebugMessenger::new(&entry, instance.clone(), enable_validation_layers);

        let surface = VulkanSurface::new(
            &entry,
            instance.clone(),
            window.raw_display_handle().unwrap(),
            window.raw_window_handle().unwrap(),
        )
        .expect("failed to create window surface!");

        // Application can't function without geometry shaders or the graphics queue family or anisotropy (we could remove anisotropy)
        let device_features = vk::PhysicalDeviceFeatures::default()
            .geometry_shader(true)
            .sampler_anisotropy(true);
        let features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let features12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .descriptor_indexing(true);

        let selector = PhysicalDeviceSelector::new(instance.clone(), surface.clone());

        let required_extensions = vec![CString::from(ash::khr::swapchain::NAME)];
        let bootstrap_physical_device = selector
            .set_required_extensions(required_extensions.clone())
            .set_required_features(device_features)
            .set_required_features_12(features12)
            .set_required_features_13(features13)
            .select()
            .expect("failed to select physical device!");

        let device = LogicalDevice::new(
            instance.clone(),
            bootstrap_physical_device.physical_device,
            &bootstrap_physical_device.queue_family_indices,
            required_extensions,
            device_features,
            features12,
            features13,
        )
        .expect("failed to create logical device!");

        let graphics_queue = unsafe {
            device.handle.get_device_queue(
                bootstrap_physical_device
                    .queue_family_indices
                    .graphics_family
                    .unwrap() as u32,
                0,
            )
        };
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.handle.clone(),
            device: device.handle.clone(),
            physical_device: bootstrap_physical_device.physical_device.clone(),
            debug_settings: Default::default(),
            buffer_device_address: features12.buffer_device_address == 1,
            allocation_sizes: Default::default(),
        })
        .expect("failed to create allocator!");

        Self {
            entry,
            window,
            instance,
            physical_device: bootstrap_physical_device.physical_device,
            device,
            queue_family_indices: bootstrap_physical_device.queue_family_indices,
            graphics_queue,
            surface,
            swapchain_support: bootstrap_physical_device.swapchain_support_details,
            debug_messenger,
            msaa_samples: bootstrap_physical_device.max_sample_count,
            allocator: Arc::new(Mutex::new(allocator)),
        }
    }

    pub fn create_swapchain(&mut self, window_width: u32, window_height: u32) -> MySwapchain {
        self.swapchain_support = SwapchainSupportDetails::new(&self.physical_device, &self.surface);

        let bootstrap_swapchain = SwapchainBuilder::new(
            self.instance.clone(),
            self.device.clone(),
            self.surface.clone(),
            self.swapchain_support.clone(),
            self.queue_family_indices,
        )
        .desired_extent(window_width, window_height)
        .desired_present_mode(vk::PresentModeKHR::FIFO)
        .desired_surface_format(
            vk::SurfaceFormatKHR::default()
                .format(vk::Format::B8G8R8A8_UNORM)
                .color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR),
        )
        .add_image_usage_flags(vk::ImageUsageFlags::TRANSFER_DST)
        .build();
        bootstrap_swapchain
    }

    pub fn create_command_pool(
        &self,
        flags: vk::CommandPoolCreateFlags,
    ) -> Result<vk::CommandPool, vk::Result> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(flags)
            .queue_family_index(self.queue_family_indices.graphics_family.unwrap() as u32);

        unsafe { self.device.handle.create_command_pool(&pool_info, None) }
    }

    pub fn create_command_buffers(
        &self,
        command_pool: vk::CommandPool,
        count: u32,
    ) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count);

        unsafe { self.device.handle.allocate_command_buffers(&alloc_info) }
    }

    pub fn create_frame_data(&self, count: usize) -> Vec<Arc<Mutex<FrameData>>> {
        (0..count)
            .map(|i| {
                let command_pool = self
                    .create_command_pool(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .expect("failed to create command pool!");
                let command_buffer = self
                    .create_command_buffers(command_pool, 1)
                    .expect("failed to create command buffer!")[0];

                let render_fence = self
                    .create_fence(vk::FenceCreateFlags::SIGNALED)
                    .expect("failed to create render fence!");
                let swapchain_semaphore = self
                    .create_semaphore(vk::SemaphoreCreateFlags::empty())
                    .expect("failed to create swapchain semaphore!");
                let render_semaphore = self
                    .create_semaphore(vk::SemaphoreCreateFlags::empty())
                    .expect("failed to create swapchain semaphore!");

                let frame_sizes = vec![
                    PoolSizeRatio::new(vk::DescriptorType::STORAGE_IMAGE, 3.),
                    PoolSizeRatio::new(vk::DescriptorType::STORAGE_BUFFER, 3.),
                    PoolSizeRatio::new(vk::DescriptorType::UNIFORM_BUFFER, 3.),
                    PoolSizeRatio::new(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 4.),
                ];
                let mut frame_descriptors = DescriptorAllocatorGrowable::new(self.device.clone());
                frame_descriptors.init(1000, frame_sizes);

                let gpu_scene_data_buffer = self.create_buffer(
                    &format!("gpuscenedata buffer {}", i),
                    size_of::<GPUSceneData>(),
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    MemoryLocation::CpuToGpu,
                );

                FrameData::new(
                    self.device.clone(),
                    command_pool,
                    command_buffer,
                    render_fence,
                    swapchain_semaphore,
                    render_semaphore,
                    frame_descriptors,
                    gpu_scene_data_buffer,
                )
            })
            .collect::<Vec<_>>()
    }

    pub fn create_fence(&self, flags: vk::FenceCreateFlags) -> Result<vk::Fence, vk::Result> {
        let fence_create_info = vk::FenceCreateInfo::default().flags(flags);
        unsafe { self.device.handle.create_fence(&fence_create_info, None) }
    }

    pub fn create_semaphore(
        &self,
        flags: vk::SemaphoreCreateFlags,
    ) -> Result<vk::Semaphore, vk::Result> {
        let create_info = vk::SemaphoreCreateInfo::default().flags(flags);
        unsafe { self.device.handle.create_semaphore(&create_info, None) }
    }

    pub fn transition_image_layout(
        &self,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_WRITE)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect_mask)
                    .base_mip_level(0)
                    .level_count(vk::REMAINING_MIP_LEVELS)
                    .base_array_layer(0)
                    .layer_count(vk::REMAINING_ARRAY_LAYERS),
            )
            .image(image);

        let barriers = [barrier];
        let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);

        unsafe {
            self.device
                .handle
                .cmd_pipeline_barrier2(command_buffer, &dependency_info)
        };
    }

    pub fn create_image(
        &mut self,
        img_extent: vk::Extent3D,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        memory_location: gpu_allocator::MemoryLocation,
        mip_levels: u32,
        num_samples: vk::SampleCountFlags,
        flags: vk::ImageCreateFlags,
        array_layers: u32,
    ) -> (vk::Image, Allocation) {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(img_extent.into())
            .mip_levels(mip_levels)
            .array_layers(array_layers)
            .format(format)
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(num_samples)
            .flags(flags);

        let image = unsafe { self.device.handle.create_image(&image_info, None) }
            .expect("failed to create image!");

        let mem_requirements = unsafe { self.device.handle.get_image_memory_requirements(image) };

        let is_linear = tiling == vk::ImageTiling::LINEAR;
        let image_allocation = self
            .allocator
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name: "",
                requirements: mem_requirements,
                location: memory_location,
                linear: is_linear,
                allocation_scheme: AllocationScheme::DedicatedImage(image),
            })
            .expect("failed to allocate image!");

        unsafe {
            self.device
                .handle
                .bind_image_memory(image, image_allocation.memory(), 0)
        }
        .expect("failed to bind image memory!");

        (image, image_allocation)
    }

    pub fn create_image_view(
        &self,
        image: vk::Image,
        format: vk::Format,
        aspect_mask: vk::ImageAspectFlags,
        mip_levels: u32,
        layer_count: u32,
    ) -> vk::ImageView {
        let image_view_type = if layer_count == 1 {
            vk::ImageViewType::TYPE_2D
        } else if layer_count == 6 {
            vk::ImageViewType::CUBE
        } else {
            panic!("not sure what to do with this yet!");
        };
        let component_mapping = vk::ComponentMapping::default();
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect_mask)
            .base_mip_level(0)
            .level_count(mip_levels)
            .base_array_layer(0)
            .layer_count(layer_count);

        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(image_view_type)
            .format(format)
            .components(component_mapping)
            .subresource_range(subresource_range);

        let image_view = unsafe {
            self.device
                .handle
                .create_image_view(&image_view_create_info, None)
        }
        .expect("failed to create image view!");
        image_view
    }

    pub fn create_allocated_image(
        &mut self,
        extent: vk::Extent3D,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        memory_location: gpu_allocator::MemoryLocation,
        mip_levels: u32,
        layer_count: u32,
        num_samples: vk::SampleCountFlags,
        flags: vk::ImageCreateFlags,
    ) -> AllocatedImage {
        let (image, allocation) = self.create_image(
            extent,
            format,
            tiling,
            usage,
            memory_location,
            mip_levels,
            num_samples,
            flags,
            layer_count,
        );

        let depth_formats = vec![
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];

        let aspect_mask = if depth_formats.contains(&format) {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let image_view =
            self.create_image_view(image, format, aspect_mask, mip_levels, layer_count);

        AllocatedImage {
            device: self.device.clone(),
            allocator: self.allocator.clone(),
            image,
            image_view,
            allocation: ManuallyDrop::new(allocation),
            extent,
            format,
        }
    }

    pub fn create_shader_module<P>(&self, path: P) -> Result<vk::ShaderModule, ShaderModuleError>
    where
        P: AsRef<Path>,
    {
        match File::open(path) {
            Ok(mut spv_file) => match read_spv(&mut spv_file) {
                Ok(shader_code) => {
                    let vertex_shader_info =
                        vk::ShaderModuleCreateInfo::default().code(&shader_code);
                    match unsafe {
                        self.device
                            .handle
                            .create_shader_module(&vertex_shader_info, None)
                    } {
                        Ok(shader_module) => Ok(shader_module),
                        Err(shader_error) => {
                            Err(ShaderModuleError::CreateShaderModuleError(shader_error))
                        }
                    }
                }
                Err(read_spv_error) => Err(ShaderModuleError::ReadSPVError(read_spv_error)),
            },
            Err(file_err) => Err(ShaderModuleError::FileError(file_err)),
        }
        /*         if let Ok(mut spv_file) = File::open(path) {
            let shader_code = read_spv(&mut spv_file).expect("Failed to read shader spv file");
            let vertex_shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
            let shader_module = unsafe {
                self.device
                    .handle
                    .create_shader_module(&vertex_shader_info, None)
                    .expect("shader module error")
            };
            Ok(shader_module)
        } else {
            E
        } */
    }

    pub fn init_pipelines(
        &mut self,
        draw_image_descriptor_layout: vk::DescriptorSetLayout,
    ) -> (Vec<ComputeEffect>, Arc<PipelineLayout>) {
        self.init_background_pipelines(draw_image_descriptor_layout)
    }

    pub fn init_background_pipelines(
        &mut self,
        draw_image_descriptor_layout: vk::DescriptorSetLayout,
    ) -> (Vec<ComputeEffect>, Arc<PipelineLayout>) {
        let push_constant = vk::PushConstantRange::default()
            .offset(0)
            .size(size_of::<ComputePushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);

        let set_layouts = [draw_image_descriptor_layout];
        let push_constant_ranges = [push_constant];
        let compute_layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges);
        /*
        let compute_effect_pipeline_layout = unsafe {
            self.device
                .handle
                .create_pipeline_layout(&compute_layout, None)
        }
        .expect("failed to create gradient pipeline layout!"); */
        let compute_effect_pipeline_layout =
            PipelineLayout::new(self.device.clone(), compute_layout_create_info)
                .expect("failed to create compute effect pipeline layout!");

        let gradient_shader = self
            .create_shader_module("shaders/gradient.comp.spv")
            .expect("failed to load shader module!");
        let sky_shader = self
            .create_shader_module("shaders/sky.comp.spv")
            .expect("failed to load shader module!");

        let shader_entry_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(gradient_shader)
            .name(shader_entry_name);

        let compute_pipeline_create_info = vk::ComputePipelineCreateInfo::default()
            .layout(compute_effect_pipeline_layout.handle)
            .stage(stage_info);

        let mut sky_pipeline_create_info = compute_pipeline_create_info.clone();
        sky_pipeline_create_info.stage.module = sky_shader;
        let create_infos = [compute_pipeline_create_info, sky_pipeline_create_info];

        let pipelines = unsafe {
            self.device.handle.create_compute_pipelines(
                vk::PipelineCache::null(),
                &create_infos,
                None,
            )
        }
        .expect("failed to create gradient pipeline!");
        let gradient_pipeline = pipelines[0];
        let sky_pipeline = pipelines[1];

        let gradient = ComputeEffect {
            name: String::from("gradient"),
            pipeline: Pipeline::new(
                self.device.clone(),
                gradient_pipeline,
                compute_effect_pipeline_layout.clone(),
            ),
            data: ComputePushConstants {
                data1: glam::Vec4::new(1., 0., 0., 1.),
                data2: glam::Vec4::new(0., 0., 1., 1.),
                data3: glam::Vec4::ZERO,
                data4: glam::Vec4::ZERO,
            },
        };

        let sky = ComputeEffect {
            name: String::from("sky"),
            pipeline: Pipeline::new(
                self.device.clone(),
                sky_pipeline,
                compute_effect_pipeline_layout.clone(),
            ),
            data: ComputePushConstants {
                data1: glam::Vec4::new(0.1, 0.2, 0.4, 0.97),
                data2: glam::Vec4::ZERO,
                data3: glam::Vec4::ZERO,
                data4: glam::Vec4::ZERO,
            },
        };

        unsafe {
            self.device
                .handle
                .destroy_shader_module(gradient_shader, None);
            self.device.handle.destroy_shader_module(sky_shader, None);
        };

        (vec![gradient, sky], compute_effect_pipeline_layout)
    }

    pub fn init_marching_cubes_pipeline(&mut self, desc_layout: &DescriptorLayout) -> Pipeline {
        let set_layouts = [desc_layout.handle];
        let push_constant_ranges = [];
        let compute_layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges);

        let compute_effect_pipeline_layout =
            PipelineLayout::new(self.device.clone(), compute_layout_create_info)
                .expect("failed to create compute effect pipeline layout!");

        let marching_cube_shader = self
            .create_shader_module("shaders/marching_cubes.comp.spv")
            .expect("failed to load shader module!");

        let shader_entry_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(marching_cube_shader)
            .name(shader_entry_name);

        let compute_pipeline_create_info = vk::ComputePipelineCreateInfo::default()
            .layout(compute_effect_pipeline_layout.handle)
            .stage(stage_info);

        let create_infos = [compute_pipeline_create_info];

        let pipelines = unsafe {
            self.device.handle.create_compute_pipelines(
                vk::PipelineCache::null(),
                &create_infos,
                None,
            )
        }
        .expect("failed to create gradient pipeline!");
        let vk_pipeline = pipelines[0];

        let pipeline = Pipeline::new(
            self.device.clone(),
            vk_pipeline,
            compute_effect_pipeline_layout,
        );

        unsafe {
            self.device
                .handle
                .destroy_shader_module(marching_cube_shader, None);
        };

        pipeline
    }

    pub fn init_descriptors(
        &mut self,
        global_descriptor_allocator: &mut DescriptorAllocatorGrowable,
        draw_image: &AllocatedImage,
    ) -> Descriptor {
        let draw_image_desc_ty = vk::DescriptorType::STORAGE_IMAGE;
        let sizes = vec![PoolSizeRatio {
            desc_type: draw_image_desc_ty,
            ratio: 1.,
        }];

        global_descriptor_allocator.init(10, sizes);

        let draw_image_descriptor_layout = DescriptorLayoutBuilder::new()
            .add_binding(0, draw_image_desc_ty)
            .build(self.device.clone(), vk::ShaderStageFlags::COMPUTE)
            .expect("failed to create draw image descriptor layout!");

        let draw_image_descriptors =
            global_descriptor_allocator.allocate(draw_image_descriptor_layout);

        let mut desc_writer = DescriptorWriter::new();

        desc_writer.write_image(
            0,
            draw_image.image_view,
            vk::Sampler::null(),
            vk::ImageLayout::GENERAL,
            vk::DescriptorType::STORAGE_IMAGE,
        );

        desc_writer.update_set(self.device.clone(), draw_image_descriptors);

        let image_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(draw_image.image_view);

        let image_infos = [image_info];
        let draw_image_write = vk::WriteDescriptorSet::default()
            .dst_binding(0)
            .dst_set(draw_image_descriptors)
            .descriptor_count(1)
            .descriptor_type(draw_image_desc_ty)
            .image_info(&image_infos);

        let desc_writes = [draw_image_write];
        let desc_copies = [];
        unsafe {
            self.device
                .handle
                .update_descriptor_sets(&desc_writes, &desc_copies)
        }

        Descriptor::new(
            self.device.clone(),
            draw_image_descriptors,
            draw_image_descriptor_layout,
        )
    }

    pub fn init_gpu_scene_descriptor_layout(&self) -> DescriptorLayout {
        let layout = DescriptorLayoutBuilder::new()
            .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER)
            .build(
                self.device.clone(),
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            )
            .expect("failed to build gpu scene desc layout!");

        DescriptorLayout::new(self.device.clone(), layout)
    }

    pub fn init_marching_cubes_descriptor_layout(&self) -> DescriptorLayout {
        let layout = DescriptorLayoutBuilder::new()
            .add_binding(0, vk::DescriptorType::STORAGE_BUFFER)
            .add_binding(1, vk::DescriptorType::STORAGE_BUFFER)
            .add_binding(2, vk::DescriptorType::STORAGE_BUFFER)
            .build(self.device.clone(), vk::ShaderStageFlags::COMPUTE)
            .expect("failed to create marching cube set layout");

        DescriptorLayout::new(self.device.clone(), layout)
    }

    pub fn init_immediate_command(&self) -> ImmediateCommand {
        let command_pool = self
            .create_command_pool(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .expect("failed to create immediate submit command pool!");

        let command_buffer = self
            .create_command_buffers(command_pool, 1)
            .expect("failed to create immediate submit command buffer!")[0];

        let fence = self
            .create_fence(vk::FenceCreateFlags::SIGNALED)
            .expect("failed to create immediate submit fence!");

        ImmediateCommand::new(self.device.clone(), fence, command_buffer, command_pool)
    }

    pub fn create_buffer(
        &self,
        name: &str,
        size: usize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> AllocatedBuffer {
        AllocatedBuffer::new(
            self.device.clone(),
            self.allocator.clone(),
            name,
            size as u64,
            usage,
            location,
        )
    }
}

pub fn find_memory_type(
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> u32 {
    let mem_properties =
        unsafe { instance.get_physical_device_memory_properties(*physical_device) };
    for i in 0..(mem_properties.memory_type_count as usize) {
        if (type_filter & (1 << i)) != 0
            && (mem_properties.memory_types[i].property_flags & properties) == properties
        {
            return i as u32;
        }
    }

    panic!("failed to find suitable memory type!");
}

//#[derive(Clone)]
pub struct FrameData {
    device: Arc<LogicalDevice>,
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub render_fence: vk::Fence,
    pub swapchain_semaphore: vk::Semaphore,
    pub render_semaphore: vk::Semaphore,
    pub frame_descriptors: DescriptorAllocatorGrowable,
    pub gpu_scene_data_buffer: AllocatedBuffer,
}

impl FrameData {
    pub fn new(
        device: Arc<LogicalDevice>,
        command_pool: vk::CommandPool,
        command_buffer: vk::CommandBuffer,
        render_fence: vk::Fence,
        swapchain_semaphore: vk::Semaphore,
        render_semaphore: vk::Semaphore,
        frame_descriptors: DescriptorAllocatorGrowable,
        gpu_scene_data_buffer: AllocatedBuffer,
    ) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            device,
            command_pool,
            command_buffer,
            render_fence,
            swapchain_semaphore,
            render_semaphore,
            frame_descriptors,
            gpu_scene_data_buffer,
        }))
    }
}

impl Drop for FrameData {
    fn drop(&mut self) {
        unsafe {
            self.device
                .handle
                .destroy_command_pool(self.command_pool, None);

            self.device.handle.destroy_fence(self.render_fence, None);
            self.device
                .handle
                .destroy_semaphore(self.render_semaphore, None);
            self.device
                .handle
                .destroy_semaphore(self.swapchain_semaphore, None);
        };
    }
}

pub struct ImmediateCommand {
    device: Arc<LogicalDevice>,
    pub fence: vk::Fence,
    pub command_buffer: vk::CommandBuffer,
    pub command_pool: vk::CommandPool,
}

impl ImmediateCommand {
    pub fn new(
        device: Arc<LogicalDevice>,
        fence: vk::Fence,
        command_buffer: vk::CommandBuffer,
        command_pool: vk::CommandPool,
    ) -> Self {
        Self {
            device,
            fence,
            command_buffer,
            command_pool,
        }
    }
}

impl Drop for ImmediateCommand {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_fence(self.fence, None);
            self.device
                .handle
                .destroy_command_pool(self.command_pool, None)
        };
    }
}
