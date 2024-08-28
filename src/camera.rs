use glam::Vec4Swizzles;
use winit::{event::KeyEvent, keyboard::Key};

pub struct Camera {
    pub velocity: glam::Vec3,
    pub position: glam::Vec3,
    pub pitch: f32,
    pub yaw: f32,
}

impl Camera {
    pub fn new(position: glam::Vec3, pitch: f32, yaw: f32) -> Self {
        Self {
            velocity: glam::Vec3::ZERO,
            position,
            pitch,
            yaw,
        }
    }

    pub fn get_view_matrix(&self) -> glam::Mat4 {
        // to create a correct model view, we need to move the world in opposite
        // direction to the camera
        //  so we will create the camera model matrix and invert
        let camera_translation = glam::Mat4::from_translation(self.position);
        let camera_rotation = self.get_rotation_matrix();
        (camera_translation * camera_rotation).inverse()
    }

    pub fn get_rotation_matrix(&self) -> glam::Mat4 {
        // fairly typical FPS style camera. we join the pitch and yaw rotations into
        // the final rotation matrix
        let pitch_rotation = glam::Quat::from_axis_angle(glam::vec3(1., 0., 0.), self.pitch);
        let yaw_rotation = glam::Quat::from_axis_angle(glam::vec3(0., -1., 0.), self.yaw);
        glam::Mat4::from_quat(yaw_rotation) * glam::Mat4::from_quat(pitch_rotation)
    }

    pub fn process_keyboard_input_event(&mut self, event: KeyEvent) {
        let pressed = event.state.is_pressed();
        if let Key::Character(ch) = event.logical_key.as_ref() {
            match ch {
                "w" => {
                    if pressed {
                        self.velocity.z = -1.;
                    } else {
                        self.velocity.z = 0.;
                    }
                }
                "s" => {
                    if pressed {
                        self.velocity.z = 1.;
                    } else {
                        self.velocity.z = 0.;
                    }
                }
                "a" => {
                    if pressed {
                        self.velocity.x = -1.;
                    } else {
                        self.velocity.x = 0.;
                    }
                }
                "d" => {
                    if pressed {
                        self.velocity.x = 1.;
                    } else {
                        self.velocity.x = 0.;
                    }
                }
                _ => (),
            }
        };
    }

    pub fn process_mouse_input_event(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw += delta_x / 200.;
        self.pitch -= delta_y / 200.;
    }

    pub fn update(&mut self) {
        let camera_rotation = self.get_rotation_matrix();
        let vel = self.velocity * 0.5;
        self.position += (camera_rotation * glam::Vec4::new(vel.x, vel.y, vel.z, 0.)).xyz();
    }
}
