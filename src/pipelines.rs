use ash::vk;
use nanoid::nanoid;
use std::{ffi::CStr, sync::Arc};

use crate::ash_bootstrap::LogicalDevice;

pub struct Pipeline {
    id: String,
    device: Arc<LogicalDevice>,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: Arc<PipelineLayout>,
}

impl Pipeline {
    pub fn new(
        device: Arc<LogicalDevice>,
        pipeline: vk::Pipeline,
        pipeline_layout: Arc<PipelineLayout>,
    ) -> Self {
        let id = nanoid!();
        Self {
            id,
            device,
            pipeline,
            pipeline_layout,
        }
    }
}

impl PartialEq for Pipeline {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_pipeline(self.pipeline, None);
        }
    }
}

pub struct PipelineLayout {
    device: Arc<LogicalDevice>,
    pub handle: vk::PipelineLayout,
}

impl PipelineLayout {
    pub fn new(
        device: Arc<LogicalDevice>,
        create_info: vk::PipelineLayoutCreateInfo,
    ) -> Result<Arc<PipelineLayout>, vk::Result> {
        let result = unsafe { device.handle.create_pipeline_layout(&create_info, None) };
        if let Ok(handle) = result {
            Ok(Arc::new(Self { device, handle }))
        } else {
            Err(result.err().unwrap())
        }
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .handle
                .destroy_pipeline_layout(self.handle, None)
        };
    }
}

#[derive(Clone)]
pub struct PipelineBuilder<'a> {
    input_assembly: vk::PipelineInputAssemblyStateCreateInfo<'a>,
    shader_stages: Vec<vk::PipelineShaderStageCreateInfo<'a>>,
    rasterizer: vk::PipelineRasterizationStateCreateInfo<'a>,
    color_blend_attachment: vk::PipelineColorBlendAttachmentState,
    multisampling: vk::PipelineMultisampleStateCreateInfo<'a>,
    pipeline_layout: Arc<PipelineLayout>,
    depth_stencil: vk::PipelineDepthStencilStateCreateInfo<'a>,
    color_attachment_format: vk::Format,
    depth_attachment_format: vk::Format,
}

impl PipelineBuilder<'_> {
    pub fn new(pipeline_layout: Arc<PipelineLayout>) -> Self {
        Self {
            shader_stages: vec![],
            input_assembly: vk::PipelineInputAssemblyStateCreateInfo::default(),
            rasterizer: vk::PipelineRasterizationStateCreateInfo::default(),
            color_blend_attachment: vk::PipelineColorBlendAttachmentState::default(),
            multisampling: vk::PipelineMultisampleStateCreateInfo::default(),
            pipeline_layout,
            depth_stencil: vk::PipelineDepthStencilStateCreateInfo::default(),
            color_attachment_format: vk::Format::default(),
            depth_attachment_format: vk::Format::default(),
        }
    }

    pub fn set_shaders(
        mut self,
        vertex_shader: vk::ShaderModule,
        fragment_shader: vk::ShaderModule,
    ) -> Self {
        let shader_entry_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

        let vertex_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader)
            .name(shader_entry_name);

        let fragment_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader)
            .name(shader_entry_name);
        self.shader_stages.push(vertex_stage);
        self.shader_stages.push(fragment_stage);
        self
    }

    pub fn set_input_topology(mut self, topology: vk::PrimitiveTopology) -> Self {
        self.input_assembly = self
            .input_assembly
            .topology(topology)
            .primitive_restart_enable(false);
        // we are not going to use primitive restart on the entire tutorial so leave it on false
        self
    }

    pub fn set_polygon_mode(mut self, mode: vk::PolygonMode) -> Self {
        self.rasterizer = self.rasterizer.polygon_mode(mode).line_width(1.);
        self
    }

    pub fn set_cull_mode(
        mut self,
        cull_mode: vk::CullModeFlags,
        front_face: vk::FrontFace,
    ) -> Self {
        self.rasterizer = self.rasterizer.cull_mode(cull_mode).front_face(front_face);
        self
    }

    pub fn set_multisampling_none(mut self) -> Self {
        self.multisampling = self
            .multisampling
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.)
            .sample_mask(&[])
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false);
        self
    }

    pub fn disable_blending(mut self) -> Self {
        self.color_blend_attachment = self
            .color_blend_attachment
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);
        self
    }

    pub fn enable_blending_additive(mut self) -> Self {
        self.color_blend_attachment = self
            .color_blend_attachment
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::DST_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);
        self
    }

    pub fn enable_blending_alphablend(mut self) -> Self {
        self.color_blend_attachment = self
            .color_blend_attachment
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE_MINUS_DST_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::DST_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);
        self
    }

    pub fn set_color_attachment_format(mut self, format: vk::Format) -> Self {
        self.color_attachment_format = format;
        self
    }

    pub fn set_depth_attachment_format(mut self, format: vk::Format) -> Self {
        self.depth_attachment_format = format;
        self
    }

    pub fn enable_depth_test(
        mut self,
        depth_write_enable: bool,
        depth_compare_op: vk::CompareOp,
    ) -> Self {
        self.depth_stencil = self
            .depth_stencil
            .depth_test_enable(true)
            .depth_write_enable(depth_write_enable)
            .depth_compare_op(depth_compare_op)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
            .front(vk::StencilOpState::default())
            .back(vk::StencilOpState::default())
            .min_depth_bounds(0.)
            .max_depth_bounds(1.);
        self
    }

    pub fn disable_depth_test(mut self) -> Self {
        self.depth_stencil = self
            .depth_stencil
            .depth_test_enable(false)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::NEVER)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
            .front(vk::StencilOpState::default())
            .back(vk::StencilOpState::default())
            .min_depth_bounds(0.)
            .max_depth_bounds(1.);
        self
    }

    pub fn build_pipeline(self, device: Arc<LogicalDevice>) -> Result<Pipeline, vk::Result> {
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let attachments = [self.color_blend_attachment];
        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&attachments);

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

        let state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&state);

        let color_attachment_formats = [self.color_attachment_format];
        let mut render_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_attachment_formats)
            .depth_attachment_format(self.depth_attachment_format);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .push_next(&mut render_info)
            .stages(&self.shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&self.input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&self.rasterizer)
            .multisample_state(&self.multisampling)
            .color_blend_state(&color_blending)
            .depth_stencil_state(&self.depth_stencil)
            .layout(self.pipeline_layout.handle)
            .dynamic_state(&dynamic_state);

        let create_infos = [pipeline_info];
        let result = unsafe {
            device
                .handle
                .create_graphics_pipelines(vk::PipelineCache::null(), &create_infos, None)
        };

        if let Ok(new_pipelines) = result {
            let new_pipeline = new_pipelines[0];
            Ok(Pipeline::new(
                device.clone(),
                new_pipeline,
                self.pipeline_layout,
            ))
        } else {
            let (_, error) = result.err().unwrap();
            Err(error)
        }
    }
}
