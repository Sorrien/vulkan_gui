use std::{
    mem::{size_of, ManuallyDrop},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use ash::vk::{self};
use ash_bootstrap::LogicalDevice;
use base_vulkan::{BaseVulkanState, FrameData};
use buffers::{copy_buffer_to_image, copy_to_cpu_buffer};
use descriptors::{Descriptor, DescriptorAllocatorGrowable, DescriptorLayout};
use gpu_allocator::{vulkan::*, MemoryLocation};
use swapchain::MySwapchain;
use vk_imgui::init_imgui;
use winit::{
    event::{DeviceEvent, Event, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

pub use imgui::*;
pub use winit::*;

pub mod base_vulkan;
pub mod buffers;
pub mod debug;
pub mod descriptors;
pub mod swapchain;
pub mod vk_imgui;

const FRAME_OVERLAP: usize = 2;

pub trait GuiApp {
    fn update(&mut self, vulkan_engine: &mut VulkanEngine, delta_time: Duration) -> bool;
    fn ui(&mut self, ui: &mut imgui::Ui, vulkan_engine: &mut VulkanEngine);
}

pub struct VulkanGuiApp {
    application_title: String,
    pub min_tick_time: u64,
    pub max_tick_time: u64,
    pub font_size: f64,
}

impl VulkanGuiApp {
    pub fn new(application_title: String) -> Self {
        Self {
            application_title,
            min_tick_time: 16,
            max_tick_time: 120,
            font_size: 13.0,
        }
    }

    pub fn run<T: GuiApp>(&mut self, app: T) -> Result<(), winit::error::EventLoopError> {
        let (event_loop, window) = VulkanEngine::init_window(3440, 1440, &self.application_title);

        let mut vulkan_engine = VulkanEngine::new(
            window,
            self.application_title.clone(),
            self.min_tick_time,
            self.max_tick_time,
        );
        vulkan_engine.init_commands();
        let (imgui, winit_platform, imgui_renderer) = vulkan_engine.init_imgui(self.font_size);

        vulkan_engine.run(event_loop, imgui, winit_platform, imgui_renderer, app)
    }
}

pub struct VulkanEngine {
    pub is_right_mouse_button_pressed: bool,
    pub is_left_mouse_button_pressed: bool,
    pub max_tick_time: u64,
    pub min_tick_time: u64,
    pub is_focused: bool,
    pub draw_extent: vk::Extent2D,
    pub render_scale: f32,
    pub resize_requested: bool,
    frame_number: usize,
    immediate_command: base_vulkan::ImmediateCommand,
    draw_image_descriptor: Descriptor,
    global_descriptor_allocator: descriptors::DescriptorAllocatorGrowable,
    draw_image: AllocatedImage,
    frames: Vec<Arc<Mutex<FrameData>>>,
    pub swapchain: MySwapchain,
    base: BaseVulkanState,
}

impl VulkanEngine {
    pub fn new(
        window: Window,
        application_title: String,
        min_tick_time: u64,
        max_tick_time: u64,
    ) -> Self {
        let mut base = BaseVulkanState::new(window, application_title);

        let window_size = base.window.inner_size();
        let window_height = window_size.height;
        let window_width = window_size.width;

        let swapchain = base.create_swapchain(window_width, window_height);

        let draw_image_extent = vk::Extent2D {
            width: window_width,
            height: window_height,
        };
        let draw_image_allocated = base.create_allocated_image(
            draw_image_extent.into(),
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            gpu_allocator::MemoryLocation::GpuOnly,
            1,
            1,
            vk::SampleCountFlags::TYPE_1,
            vk::ImageCreateFlags::empty(),
        );

        let mut global_descriptor_allocator = DescriptorAllocatorGrowable::new(base.device.clone());

        let draw_image_descriptor =
            base.init_descriptors(&mut global_descriptor_allocator, &draw_image_allocated);

        let immediate_command = base.init_immediate_command();

        Self {
            base,
            frames: vec![],
            frame_number: 0,
            draw_image: draw_image_allocated,
            swapchain,
            global_descriptor_allocator,
            draw_image_descriptor,
            immediate_command,
            resize_requested: false,
            draw_extent: draw_image_extent,
            render_scale: 1.,
            is_focused: false,
            min_tick_time,
            max_tick_time,
            is_right_mouse_button_pressed: false,
            is_left_mouse_button_pressed: false,
        }
    }

    pub fn init_imgui(
        &self,
        font_size: f64,
    ) -> (
        imgui::Context,
        imgui_winit_support::WinitPlatform,
        imgui_rs_vulkan_renderer::Renderer,
    ) {
        init_imgui(
            &self.base.window,
            &self.base.allocator,
            &self.base.device,
            self.base.graphics_queue,
            self.immediate_command.command_pool,
            self.swapchain.format,
            font_size,
        )
    }

    pub fn init_commands(&mut self) {
        self.frames = self.base.create_frame_data(FRAME_OVERLAP);
    }

    pub fn init_window(
        width: u32,
        height: u32,
        title: &str,
    ) -> (EventLoop<()>, winit::window::Window) {
        let event_loop = EventLoop::new().unwrap();
        let window = WindowBuilder::new()
            .with_title(title)
            .with_inner_size(winit::dpi::LogicalSize::new(width, height))
            .build(&event_loop)
            .unwrap();

        (event_loop, window)
    }

    pub fn get_current_frame(&self) -> Arc<Mutex<FrameData>> {
        self.frames[self.frame_number % FRAME_OVERLAP].clone()
    }

    pub fn run<T: GuiApp>(
        &mut self,
        event_loop: EventLoop<()>,
        imgui: imgui::Context,
        platform: imgui_winit_support::WinitPlatform,
        imgui_renderer: imgui_rs_vulkan_renderer::Renderer,
        app: T,
    ) -> Result<(), winit::error::EventLoopError> {
        self.main_loop(event_loop, imgui, platform, imgui_renderer, app)
    }

    fn main_loop<T: GuiApp>(
        &mut self,
        event_loop: EventLoop<()>,
        mut imgui: imgui::Context,
        mut platform: imgui_winit_support::WinitPlatform,
        mut imgui_renderer: imgui_rs_vulkan_renderer::Renderer,
        mut app: T,
    ) -> Result<(), winit::error::EventLoopError> {
        let mut last_frame = Instant::now();
        let mut delta_time = Instant::now() - last_frame;
        let min_tick_time = Duration::from_millis(self.min_tick_time);
        let max_tick_time = Duration::from_millis(self.max_tick_time);
        let mut total_time_since_last_tick = delta_time;
        let mut num_ticks = 0;

        let mut input_counter = 0;

        event_loop.run(move |event, elwt| {
            platform.handle_event(imgui.io_mut(), &self.base.window, &event);

            match event {
                Event::NewEvents(_) => {
                    let now = Instant::now();
                    delta_time = now - last_frame;

                    total_time_since_last_tick += delta_time;
                    imgui.io_mut().update_delta_time(delta_time);
                    last_frame = now;
                }
                Event::WindowEvent {
                    window_id: _,
                    event: WindowEvent::Resized(_new_size),
                } => {
                    self.resize_requested = true;
                }
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    unsafe { self.base.device.handle.device_wait_idle() }
                        .expect("failed to wait for idle on exit!");
                    elwt.exit()
                }
                Event::AboutToWait => {
                    //AboutToWait is the new MainEventsCleared
                    self.base.window.request_redraw();
                }
                Event::WindowEvent {
                    event: WindowEvent::RedrawRequested,
                    window_id: _,
                } => {
                    let size = self.base.window.inner_size();

                    if self.resize_requested {
                        if size.width > 0 && size.height > 0 {
                            self.resize_swapchain();
                        } else {
                            return;
                        }
                    }
                    let is_updated: bool = app.update(self, delta_time);

                    if (total_time_since_last_tick >= min_tick_time
                        && (is_updated || input_counter > 0))
                        || total_time_since_last_tick >= max_tick_time
                    {
                        platform
                            .prepare_frame(imgui.io_mut(), &self.base.window)
                            .expect("failed to prepare frame!");
                        let ui = imgui.frame();

                        app.ui(ui, self);

                        platform.prepare_render(ui, &self.base.window);

                        let draw_data = imgui.render();

                        total_time_since_last_tick = Duration::from_micros(0);
                        num_ticks += 1;
                        input_counter = 0;

                        //don't attempt to draw a frame in window size is 0
                        if size.height > 0 && size.width > 0 {
                            self.draw(draw_data, &mut imgui_renderer);
                        }
                    }
                }
                Event::WindowEvent {
                    window_id: _,
                    event: WindowEvent::Focused(is_focused),
                } => {
                    self.is_focused = is_focused;
                }
                Event::DeviceEvent {
                    device_id,
                    event:
                        DeviceEvent::MouseMotion {
                            delta: (_delta_x, _delta_y),
                        },
                } => {
                    if self.is_focused
                        && (self.is_left_mouse_button_pressed || self.is_right_mouse_button_pressed)
                    {
                        input_counter += 1;
                    }
                }
                Event::DeviceEvent {
                    device_id,
                    event: DeviceEvent::MouseWheel { delta: _ },
                } => {
                    if self.is_focused {
                        input_counter += 1;
                    }
                }
                Event::DeviceEvent {
                    device_id,
                    event: DeviceEvent::Button { button, state },
                } => {
                    if self.is_focused {
                        if button == 0 {
                            self.is_left_mouse_button_pressed = state.is_pressed();
                        } else if button == 1 {
                            self.is_right_mouse_button_pressed = state.is_pressed();
                        }
                        input_counter += 1;
                    }
                }
                Event::WindowEvent {
                    window_id,
                    event:
                        WindowEvent::KeyboardInput {
                            device_id,
                            event,
                            is_synthetic,
                        },
                } => {
                    if self.is_focused {
                        input_counter += 1;
                    }
                }
                _ => (),
            }
        })
    }

    pub fn draw(
        &mut self,
        draw_data: &imgui::DrawData,
        imgui_renderer: &mut imgui_rs_vulkan_renderer::Renderer,
    ) {
        let current_frame = self.get_current_frame();
        let mut current_frame = current_frame.lock().unwrap();
        //self.update_scene();

        let fences = [current_frame.render_fence];
        unsafe {
            self.base
                .device
                .handle
                .wait_for_fences(&fences, true, u64::MAX)
        }
        .expect("failed to wait for render fence!");
        unsafe { self.base.device.handle.reset_fences(&fences) }
            .expect("failed to reset render fence!");

        current_frame.frame_descriptors.destroy_pools();
        current_frame.frame_descriptors.clear_pools();

        let swapchain = &self.swapchain;

        self.draw_extent.height = (swapchain.extent.height.min(self.draw_image.extent.height)
            as f32
            * self.render_scale) as u32;
        self.draw_extent.width = (swapchain.extent.width.min(self.draw_image.extent.width) as f32
            * self.render_scale) as u32;

        //acquire next swapchain image
        let (swapchain_image_index, _) = unsafe {
            let result = swapchain.swapchain_loader.acquire_next_image(
                swapchain.swapchain,
                u64::MAX,
                current_frame.swapchain_semaphore,
                vk::Fence::null(),
            );

            match result {
                Ok((image_index, was_next_image_acquired)) => {
                    (image_index, was_next_image_acquired)
                }
                Err(vk_result) => match vk_result {
                    vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => {
                        self.resize_requested = true;
                        return;
                    }
                    _ => panic!("failed to acquire next swapchain image!"),
                },
            }
        };

        let cmd = current_frame.command_buffer;

        unsafe {
            self.base
                .device
                .handle
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
        }
        .expect("failed to reset command buffer!");

        unsafe {
            self.base.device.handle.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .expect("failed to begin command buffer!");

        let swapchain_image = swapchain.swapchain_images[swapchain_image_index as usize];

        self.base.transition_image_layout(
            cmd,
            swapchain_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        self.base.transition_image_layout(
            cmd,
            self.draw_image.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );

        copy_image_to_image(
            self.base.device.clone(),
            cmd,
            self.draw_image.image,
            swapchain_image,
            self.draw_extent.into(),
            swapchain.extent.into(),
        );

        self.base.transition_image_layout(
            cmd,
            swapchain_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        self.draw_imgui(
            cmd,
            draw_data,
            imgui_renderer,
            swapchain.swapchain_image_views[swapchain_image_index as usize],
        );

        self.base.transition_image_layout(
            cmd,
            swapchain_image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );

        unsafe { self.base.device.handle.end_command_buffer(cmd) }
            .expect("failed to end command buffer!");

        let command_info = vk::CommandBufferSubmitInfo::default()
            .command_buffer(current_frame.command_buffer)
            .device_mask(0);
        let wait_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(current_frame.swapchain_semaphore)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT_KHR)
            .device_index(0)
            .value(1);

        let signal_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(current_frame.render_semaphore)
            .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
            .device_index(0)
            .value(1);
        let command_buffer_infos = [command_info];
        let signal_semaphore_infos = [signal_info];
        let wait_semaphore_infos = [wait_info];

        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_semaphore_infos)
            .signal_semaphore_infos(&signal_semaphore_infos)
            .command_buffer_infos(&command_buffer_infos);
        let submits = [submit];
        unsafe {
            self.base.device.handle.queue_submit2(
                self.base.graphics_queue,
                &submits,
                current_frame.render_fence,
            )
        }
        .expect("queue command submit failed!");

        let swapchains = [swapchain.swapchain];
        let render_semaphores = [current_frame.render_semaphore];
        let swapchain_image_indices = [swapchain_image_index];
        let present_info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .wait_semaphores(&render_semaphores)
            .image_indices(&swapchain_image_indices);

        let present_result = unsafe {
            swapchain
                .swapchain_loader
                .queue_present(self.base.graphics_queue, &present_info)
        };
        match present_result {
            Ok(_) => (),
            Err(vk_result) => match vk_result {
                vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => {
                    self.resize_requested = true;
                }
                _ => panic!("failed to present swap chain image!"),
            },
        }

        self.frame_number += 1;
    }

    fn resize_swapchain(&mut self) {
        unsafe { self.base.device.handle.device_wait_idle() }
            .expect("failed to wait for idle when recreating swapchain!");

        let window_size = self.base.window.inner_size();
        let window_height = window_size.height;
        let window_width = window_size.width;

        self.swapchain.destroy();
        self.swapchain = self.base.create_swapchain(window_width, window_height);
        self.resize_requested = false;
    }

    pub fn semaphore_submit_info(
        stage_mask: vk::PipelineStageFlags2,
        semaphore: vk::Semaphore,
    ) -> vk::SemaphoreSubmitInfo<'static> {
        vk::SemaphoreSubmitInfo::default()
            .semaphore(semaphore)
            .stage_mask(stage_mask)
            .device_index(0)
            .value(1)
    }

    pub fn command_buffer_submit_info(
        command_buffer: vk::CommandBuffer,
    ) -> vk::CommandBufferSubmitInfo<'static> {
        vk::CommandBufferSubmitInfo::default()
            .command_buffer(command_buffer)
            .device_mask(0)
    }
    pub fn submit_info<'a>(
        command_buffer_infos: &'a [vk::CommandBufferSubmitInfo<'a>],
        signal_semaphore_infos: &'a [vk::SemaphoreSubmitInfo<'a>],
        wait_semaphore_infos: &'a [vk::SemaphoreSubmitInfo<'a>],
    ) -> vk::SubmitInfo2<'a> {
        vk::SubmitInfo2::default()
            .wait_semaphore_infos(wait_semaphore_infos)
            .signal_semaphore_infos(signal_semaphore_infos)
            .command_buffer_infos(command_buffer_infos)
    }

    pub fn draw_imgui(
        &self,
        cmd: vk::CommandBuffer,
        draw_data: &imgui::DrawData,
        imgui_renderer: &mut imgui_rs_vulkan_renderer::Renderer,
        target_image_view: vk::ImageView,
    ) {
        let color_attachment =
            Self::attachment_info(target_image_view, None, vk::ImageLayout::GENERAL);
        let color_attachments = [color_attachment];
        let rendering_info = vk::RenderingInfo::default()
            .render_area(self.swapchain.extent.into())
            .color_attachments(&color_attachments)
            .flags(vk::RenderingFlags::CONTENTS_INLINE_EXT)
            .layer_count(1);

        unsafe {
            self.base
                .device
                .handle
                .cmd_begin_rendering(cmd, &rendering_info)
        };

        imgui_renderer
            .cmd_draw(cmd, draw_data)
            .expect("failed to draw imgui data!");

        unsafe { self.base.device.handle.cmd_end_rendering(cmd) };
    }

    fn attachment_info(
        view: vk::ImageView,
        clear: Option<vk::ClearValue>,
        layout: vk::ImageLayout,
    ) -> vk::RenderingAttachmentInfo<'static> {
        let load_op = if clear.is_some() {
            vk::AttachmentLoadOp::CLEAR
        } else {
            vk::AttachmentLoadOp::LOAD
        };
        let mut result = vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(layout)
            .load_op(load_op)
            .store_op(vk::AttachmentStoreOp::STORE);

        if let Some(clear) = clear {
            result = result.clear_value(clear);
        }

        result
    }

    pub fn immediate_submit<F: FnOnce(vk::CommandBuffer)>(&self, f: F) {
        let fences = [self.immediate_command.fence];
        let cmd = self.immediate_command.command_buffer;
        let device = &self.base.device.handle;

        unsafe { device.reset_fences(&fences) }.expect("failed to reset immediate submit fence!");
        unsafe { device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()) }
            .expect("failed to reset imm submit cmd buffer!");

        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .expect("failed to end imm submit cmd buffer!");

        f(cmd);

        unsafe { device.end_command_buffer(cmd) }.expect("failed to end imm submit cmd buffer!");

        let cmd_info = Self::command_buffer_submit_info(cmd);
        let cmd_infos = [cmd_info];
        let submit = Self::submit_info(&cmd_infos, &[], &[]);

        //we may want to find a different queue than graphics for this if possible
        let submits = [submit];
        unsafe {
            device.queue_submit2(
                self.base.graphics_queue,
                &submits,
                self.immediate_command.fence,
            )
        }
        .expect("failed to submit imm cmd!");

        let fences = [self.immediate_command.fence];
        unsafe { device.wait_for_fences(&fences, true, u64::MAX) }
            .expect("failed to wait for imm submit fence!");
    }

    pub fn create_allocated_texture_image<T>(
        &mut self,
        data: &[T],
        extent: vk::Extent3D,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        memory_location: gpu_allocator::MemoryLocation,
        mip_levels: u32,
        num_samples: vk::SampleCountFlags,
    ) -> AllocatedImage
    where
        T: Copy,
    {
        let data_size = size_of::<T>() * data.len(); //extent.depth * extent.width * extent.height;
        let upload_buffer = self.base.create_buffer(
            "texture upload buffer",
            data_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        );

        copy_to_cpu_buffer(&upload_buffer, data_size as u64, &data);

        let allocated_image = self.base.create_allocated_image(
            extent,
            format,
            tiling,
            usage | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
            memory_location,
            mip_levels,
            1,
            num_samples,
            vk::ImageCreateFlags::empty(),
        );

        self.immediate_submit(|cmd| {
            self.base.transition_image_layout(
                cmd,
                allocated_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            copy_buffer_to_image(
                cmd,
                self.base.device.clone(),
                upload_buffer.buffer,
                allocated_image.image,
                extent,
            );

            self.base.transition_image_layout(
                cmd,
                allocated_image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        });

        allocated_image
    }
}

impl Drop for VulkanEngine {
    fn drop(&mut self) {
        unsafe { self.base.device.handle.device_wait_idle() }
            .expect("failed to wait for device idle!");
    }
}

pub struct Sampler {
    device: Arc<LogicalDevice>,
    pub handle: vk::Sampler,
}

impl Sampler {
    fn new(handle: vk::Sampler, device: Arc<LogicalDevice>) -> Self {
        Self { handle, device }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_sampler(self.handle, None) };
    }
}

pub struct AllocatedImage {
    device: Arc<LogicalDevice>,
    pub allocator: Arc<Mutex<Allocator>>,
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: ManuallyDrop<Allocation>,
    pub extent: vk::Extent3D,
    pub format: vk::Format,
}

impl Drop for AllocatedImage {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_image_view(self.image_view, None);
            self.device.handle.destroy_image(self.image, None);
        }
        let allocation = unsafe { ManuallyDrop::take(&mut self.allocation) };
        self.allocator
            .lock()
            .unwrap()
            .free(allocation)
            .expect("failed to free memory for allocated image!");
    }
}

pub fn copy_image_to_image(
    device: Arc<LogicalDevice>,
    cmd: vk::CommandBuffer,
    src: vk::Image,
    dst: vk::Image,
    src_size: vk::Extent3D,
    dst_size: vk::Extent3D,
) {
    let blit_region = vk::ImageBlit2::default()
        .src_offsets([
            vk::Offset3D::default(),
            vk::Offset3D {
                x: src_size.width as i32,
                y: src_size.height as i32,
                z: src_size.depth as i32,
            },
        ])
        .dst_offsets([
            vk::Offset3D::default(),
            vk::Offset3D {
                x: dst_size.width as i32,
                y: dst_size.height as i32,
                z: dst_size.depth as i32,
            },
        ])
        .src_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .layer_count(1)
                .mip_level(0),
        )
        .dst_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .layer_count(1)
                .mip_level(0),
        );

    let regions = [blit_region];
    let blit_info = vk::BlitImageInfo2::default()
        .dst_image(dst)
        .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_image(src)
        .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .filter(vk::Filter::LINEAR)
        .regions(&regions);

    unsafe { device.handle.cmd_blit_image2(cmd, &blit_info) };
}

/// Return a `&[u8]` for any sized object passed in.
pub unsafe fn any_as_u8_slice<T: Sized>(any: &T) -> &[u8] {
    let ptr = (any as *const T) as *const u8;
    std::slice::from_raw_parts(ptr, std::mem::size_of::<T>())
}
