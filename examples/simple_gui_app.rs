use vulkan_gui::{GuiApp, VulkanEngine, VulkanGuiApp};

fn main() -> Result<(), winit::error::EventLoopError> {
    let title = String::from("Example Vulkan Application");
    let mut vulkan_gui_app = VulkanGuiApp::new(title);
    let app = MyGuiApp::new();
    vulkan_gui_app.run(app)
}

pub struct MyGuiApp {
    pub scale_float: f32,
    pub input_string: String,
}
impl MyGuiApp {
    pub fn new() -> Self {
        Self { scale_float: 1.0, input_string: String::new() }
    }
}

impl GuiApp for MyGuiApp {
    fn ui(&mut self, ui: &mut imgui::Ui, _vulkan_engine: &mut VulkanEngine) {
        ui.window("background")
            .size([500.0, 200.0], imgui::Condition::FirstUseEver)
            .build(|| {
                ui.slider("Test Scale", 0.3, 1., &mut self.scale_float);
                ui.input_text_multiline("Test Input", &mut self.input_string, ui.content_region_avail()).build();
            });
    }
    
    fn update(&mut self, _vulkan_engine: &mut VulkanEngine, _delta_time: std::time::Duration) -> bool {
        false
    }
}
