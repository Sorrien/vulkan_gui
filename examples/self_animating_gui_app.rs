use vulkan_gui::{GuiApp, VulkanEngine, VulkanGuiApp};

fn main() -> Result<(), winit::error::EventLoopError> {
    let title = String::from("Example Vulkan Application");
    let mut vulkan_gui_app = VulkanGuiApp::new(title);
    let app = MyGuiApp::new();
    vulkan_gui_app.run(app)
}

pub struct MyGuiApp {
    pub scale_float: f32,
    pub always_update: bool,
}
impl MyGuiApp {
    pub fn new() -> Self {
        Self {
            scale_float: 1.0,
            always_update: false,
        }
    }
}

impl GuiApp for MyGuiApp {
    fn ui(&mut self, ui: &mut imgui::Ui, vulkan_engine: &mut VulkanEngine) {
        ui.window("background")
            .size([500.0, 200.0], imgui::Condition::FirstUseEver)
            .build(|| {
                ui.slider("Test Scale", 0.3, 1., &mut self.scale_float);
                ui.checkbox("Enable automatic updates", &mut self.always_update);
            });
    }

    fn update(
        &mut self,
        vulkan_engine: &mut VulkanEngine,
        delta_time: std::time::Duration,
    ) -> bool {
        if self.always_update {
            if self.scale_float >= 1. {
                self.scale_float = 0.3;
            } else {
                self.scale_float += 0.01;
            }
            true
        } else {
            false
        }
    }
}
