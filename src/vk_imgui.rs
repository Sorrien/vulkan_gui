use std::sync::{Arc, Mutex};

use crate::FRAME_OVERLAP;
use ash::vk;
use ash_bootstrap::LogicalDevice;
use clipboard_rs::{Clipboard, ClipboardContext};
use gpu_allocator::vulkan::Allocator;
use imgui::{ClipboardBackend, FontConfig};
use imgui_rs_vulkan_renderer::DynamicRendering;
use winit::window::Window;

pub fn init_imgui(
    window: &Window,
    allocator: &Arc<Mutex<Allocator>>,
    device: &Arc<LogicalDevice>,
    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    format: vk::Format,
    font_size: f64,
) -> (
    imgui::Context,
    imgui_winit_support::WinitPlatform,
    imgui_rs_vulkan_renderer::Renderer,
) {
    let mut imgui = imgui::Context::create();

    imgui.set_ini_filename(None);

    let mut platform = imgui_winit_support::WinitPlatform::init(&mut imgui);
    let dpi_mode = imgui_winit_support::HiDpiMode::Rounded;

    let hidpi_factor = platform.hidpi_factor();
    let font_size = (font_size * hidpi_factor) as f32;

    imgui
        .fonts()
        .add_font(&[imgui::FontSource::DefaultFontData {
            config: Some(FontConfig {
                size_pixels: font_size,
                ..FontConfig::default()
            }),
        }]);

    imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;
    platform.attach_window(imgui.io_mut(), window, dpi_mode);
    imgui.set_clipboard_backend(VKEngineClipboard::new());

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

pub struct VKEngineClipboard {
    ctx: ClipboardContext,
}

impl VKEngineClipboard {
    pub fn new() -> Self {
        Self {
            ctx: ClipboardContext::new().unwrap(),
        }
    }
}

impl ClipboardBackend for VKEngineClipboard {
    fn get(&mut self) -> Option<String> {
        match self.ctx.get_text() {
            Ok(text) => Some(text),
            Err(err) => {
                println!("Clipboard error: {}", err);
                None
            }
        }
    }

    fn set(&mut self, value: &str) {
        match self.ctx.set_text(value.to_string()) {
            Ok(_) => (),
            Err(err) => println!("Clipboard error: {}", err),
        }
    }
}
