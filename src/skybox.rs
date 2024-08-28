use std::{mem::size_of, sync::Arc};

use ash::vk;

use crate::{
    ash_bootstrap::LogicalDevice,
    buffers::{AllocatedBuffer, GPUDrawPushConstants},
    descriptors::{
        DescriptorAllocatorGrowable, DescriptorLayout, DescriptorLayoutBuilder, DescriptorWriter,
    },
    pipelines::{Pipeline, PipelineBuilder, PipelineLayout},
    AllocatedImage, MaterialInstance, MaterialPass, MaterialResources, Sampler, VulkanEngine,
};

pub struct Skybox {
    pub pipeline: Arc<Pipeline>,
    pub material_layout: DescriptorLayout,
}

impl Skybox {
    pub fn new(engine: &VulkanEngine) -> Self {
        let material_layout = DescriptorLayoutBuilder::new()
            .add_binding(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .build(
                engine.base.device.clone(),
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            )
            .expect("failed to create material set layout");

        let pipeline = Self::build_pipeline(engine, material_layout);

        Self {
            pipeline: Arc::new(pipeline),
            material_layout: DescriptorLayout::new(engine.base.device.clone(), material_layout),
        }
    }

    pub fn build_pipeline(
        engine: &VulkanEngine,
        material_layout: vk::DescriptorSetLayout,
    ) -> Pipeline {
        let frag_shader = engine
            .base
            .create_shader_module("shaders/skybox.frag.spv")
            .expect("failed to load shader module!");
        let vert_shader = engine
            .base
            .create_shader_module("shaders/skybox.vert.spv")
            .expect("failed to load shader module!");

        let matrix_range = vk::PushConstantRange::default()
            .offset(0)
            .size(size_of::<GPUDrawPushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let layouts = [
            engine.gpu_scene_data_descriptor_layout.handle,
            material_layout,
        ];

        let push_constant_ranges = [matrix_range];
        let mesh_layout_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(&push_constant_ranges)
            .set_layouts(&layouts);

        let new_layout = PipelineLayout::new(engine.base.device.clone(), mesh_layout_info)
            .expect("failed to create pipeline layout!");

        let pipeline_builder = PipelineBuilder::new(new_layout)
            .set_shaders(vert_shader, frag_shader)
            .set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .set_polygon_mode(vk::PolygonMode::FILL)
            .set_cull_mode(vk::CullModeFlags::FRONT, vk::FrontFace::COUNTER_CLOCKWISE)
            .set_multisampling_none()
            .set_color_attachment_format(engine.draw_image.format)
            .set_depth_attachment_format(engine.depth_image.format)
            .disable_blending()
            .disable_depth_test();

        let pipeline = pipeline_builder
            .build_pipeline(engine.base.device.clone())
            .expect("failed to build skybox pipeline!");

        unsafe {
            engine
                .base
                .device
                .handle
                .destroy_shader_module(vert_shader, None);
            engine
                .base
                .device
                .handle
                .destroy_shader_module(frag_shader, None);
        }

        pipeline
    }

    pub fn clear_resources() {}

    pub fn write_material(
        &self,
        device: Arc<LogicalDevice>,
        skybox_image: &AllocatedImage,
        skybox_image_sampler: &Sampler,
        descriptor_allocator: &mut DescriptorAllocatorGrowable,
    ) -> MaterialInstance {
        let material_set = descriptor_allocator.allocate(self.material_layout.handle);

        let mut desc_writer = DescriptorWriter::new();
        desc_writer.write_image(
            1,
            skybox_image.image_view,
            skybox_image_sampler.handle,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );

        desc_writer.update_set(device.clone(), material_set);

        let mat_data = MaterialInstance {
            pipeline: self.pipeline.clone(),
            material_set,
            pass_type: MaterialPass::MainColor,
        };

        mat_data
    }
}
