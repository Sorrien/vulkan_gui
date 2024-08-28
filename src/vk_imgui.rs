use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::Allocator;
use imgui_rs_vulkan_renderer::DynamicRendering;
use winit::window::Window;

use crate::{ash_bootstrap::LogicalDevice, FRAME_OVERLAP};

pub struct ImguiContext {
    pub renderer: imgui_rs_vulkan_renderer::Renderer,
    pub platform: imgui_winit_support::WinitPlatform,
    pub imgui: imgui::Context,
}

impl ImguiContext {
    pub fn new(
        window: &Window,
        allocator: &Arc<Mutex<Allocator>>,
        device: &Arc<LogicalDevice>,
        graphics_queue: vk::Queue,
        command_pool: vk::CommandPool,
        format: vk::Format,
    ) -> Self {
        let mut imgui = imgui::Context::create();

        imgui.set_ini_filename(None);

        let mut platform = imgui_winit_support::WinitPlatform::init(&mut imgui);
        let dpi_mode = imgui_winit_support::HiDpiMode::Default;

        platform.attach_window(imgui.io_mut(), window, dpi_mode);
        imgui
            .fonts()
            .add_font(&[imgui::FontSource::DefaultFontData { config: None }]);

        let renderer = imgui_rs_vulkan_renderer::Renderer::with_gpu_allocator(
            allocator.clone(),
            device.handle.clone(),
            graphics_queue,
            command_pool,
            DynamicRendering {
                color_attachment_format: format,
                depth_attachment_format: None,
            },
            &mut imgui,
            Some(imgui_rs_vulkan_renderer::Options {
                in_flight_frames: FRAME_OVERLAP,
                ..Default::default()
            }),
        )
        .unwrap();

        Self {
            renderer,
            platform,
            imgui,
        }
    }
}

pub fn init_imgui(
    window: &Window,
    allocator: &Arc<Mutex<Allocator>>,
    device: &Arc<LogicalDevice>,
    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    format: vk::Format,
) -> (
    imgui::Context,
    imgui_winit_support::WinitPlatform,
    imgui_rs_vulkan_renderer::Renderer,
) {
    let mut imgui = imgui::Context::create();

    imgui.set_ini_filename(None);

    let mut platform = imgui_winit_support::WinitPlatform::init(&mut imgui);
    let dpi_mode = imgui_winit_support::HiDpiMode::Default;

    platform.attach_window(imgui.io_mut(), window, dpi_mode);
    imgui
        .fonts()
        .add_font(&[imgui::FontSource::DefaultFontData { config: None }]);

    let renderer = imgui_rs_vulkan_renderer::Renderer::with_gpu_allocator(
        allocator.clone(),
        device.handle.clone(),
        graphics_queue,
        command_pool,
        DynamicRendering {
            color_attachment_format: format,
            depth_attachment_format: None,
        },
        &mut imgui,
        Some(imgui_rs_vulkan_renderer::Options {
            in_flight_frames: FRAME_OVERLAP,
            ..Default::default()
        }),
    )
    .unwrap();

    (imgui, platform, renderer)
}
