use std::{
    ffi::{c_char, c_void, CStr, CString},
    sync::Arc,
};

use ash::{
    ext::debug_utils::{self},
    khr::surface::Instance as Surface,
    vk, Entry,
};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::{debug, swapchain::SwapchainSupportDetails};

pub struct VulkanSurface {
    pub loader: Surface,
    pub handle: vk::SurfaceKHR,
}

impl VulkanSurface {
    pub fn new(
        entry: &Entry,
        instance: Arc<Instance>,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> std::result::Result<Arc<VulkanSurface>, ash::vk::Result> {
        let surface_loader = Surface::new(&entry, &instance.handle);
        let surface_result = unsafe {
            ash_window::create_surface(
                &entry,
                &instance.handle,
                display_handle,
                window_handle,
                None,
            )
        };

        if let Ok(surface) = surface_result {
            Ok(Arc::new(Self {
                loader: surface_loader,
                handle: surface,
            }))
        } else {
            Err(surface_result.err().unwrap())
        }
    }
}

impl Drop for VulkanSurface {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_surface(self.handle, None);
        }
    }
}

pub struct Instance {
    pub handle: ash::Instance,
}

impl Instance {
    fn new(instance_handle: ash::Instance) -> Arc<Self> {
        Arc::new(Self {
            handle: instance_handle,
        })
    }
    pub fn builder() -> InstanceBuilder {
        InstanceBuilder::new()
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe { self.handle.destroy_instance(None) };
    }
}

pub struct InstanceBuilder {
    entry: Option<ash::Entry>,
    enable_validation_layers: bool,
    raw_display_handle: Option<RawDisplayHandle>,
    api_version: u32,
    application_name: String,
    application_version: u32,
    engine_name: String,
    engine_version: u32,
}

impl InstanceBuilder {
    pub fn new() -> Self {
        Self {
            entry: None,
            enable_validation_layers: false,
            raw_display_handle: None,
            application_name: String::from("Vulkan Application"),
            application_version: vk::make_api_version(0, 1, 0, 0),
            engine_name: String::from("No Engine"),
            engine_version: vk::make_api_version(0, 1, 0, 0),
            api_version: vk::make_api_version(0, 1, 0, 0),
        }
    }

    pub fn entry(mut self, entry: ash::Entry) -> Self {
        self.entry = Some(entry);
        self
    }

    pub fn enable_validation_layers(mut self, enable_validation_layers: bool) -> Self {
        self.enable_validation_layers = enable_validation_layers;
        self
    }

    pub fn raw_display_handle(mut self, raw_display_handle: RawDisplayHandle) -> Self {
        self.raw_display_handle = Some(raw_display_handle);
        self
    }

    pub fn api_version(mut self, api_version: u32) -> Self {
        self.api_version = api_version;
        self
    }

    pub fn application_name(mut self, application_name: String) -> Self {
        self.application_name = application_name;
        self
    }

    pub fn application_version(mut self, application_version: u32) -> Self {
        self.application_version = application_version;
        self
    }

    pub fn engine_name(mut self, engine_name: String) -> Self {
        self.engine_name = engine_name;
        self
    }

    pub fn engine_version(mut self, engine_version: u32) -> Self {
        self.engine_version = engine_version;
        self
    }

    pub fn build(self) -> Arc<Instance> {
        let app_name_cstring = CString::new(self.application_name.as_str()).unwrap();
        let app_name = app_name_cstring.as_c_str();
        let engine_name_cstring = CString::new(self.engine_name.as_str()).unwrap();
        let engine_name = engine_name_cstring.as_c_str();

        let entry = &self.entry.expect("no entry was provided!");

        let mut required_extension_names = vec![];

        if let Some(raw_display_handle) = self.raw_display_handle {
            let mut window_extension_names =
                ash_window::enumerate_required_extensions(raw_display_handle)
                    .unwrap()
                    .to_vec();

            required_extension_names.append(&mut window_extension_names);
        }

        if self.enable_validation_layers {
            required_extension_names.push(debug_utils::NAME.as_ptr());

            println!("Validation Layers enabled!");
        }

        let extension_properties = unsafe { entry.enumerate_instance_extension_properties(None) }
            .expect("failed to enumerate instance extension props!");

        println!("Enabled extensions:");
        for extension_name in required_extension_names.iter() {
            let str = unsafe { CStr::from_ptr(*extension_name) }
                .to_str()
                .expect("failed to get ext name str");

            if extension_properties.iter().any(|prop| {
                unsafe { CStr::from_ptr(prop.extension_name.as_ptr()) }
                    .to_str()
                    .unwrap()
                    == str
            }) {
                println!("{}", str);
            } else {
                panic!("required extensions were not available!");
            }
        }
        println!("");

        let appinfo = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(self.application_version)
            .engine_name(engine_name)
            .engine_version(self.engine_version)
            .api_version(self.api_version);

        let layer_names = unsafe {
            [CStr::from_bytes_with_nul_unchecked(
                b"VK_LAYER_KHRONOS_validation\0",
            )]
        };

        let layers_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&appinfo)
            .enabled_extension_names(&required_extension_names);

        let debug_info = debug::create_debug_info();
        if self.enable_validation_layers {
            Self::check_validation_layer_support(entry, &layers_names_raw);
            create_info = create_info.enabled_layer_names(&layers_names_raw);

            create_info.p_next =
                &debug_info as *const vk::DebugUtilsMessengerCreateInfoEXT as *const c_void;
        }

        let instance_handle = unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Instance creation error!")
        };

        Instance::new(instance_handle)
    }

    fn check_validation_layer_support(entry: &ash::Entry, layers_names_raw: &Vec<*const c_char>) {
        let available_layers = unsafe { entry.enumerate_instance_layer_properties() }
            .expect("failed to get available layers!");

        for name in layers_names_raw.iter() {
            let str = unsafe { CStr::from_ptr(*name) }
                .to_str()
                .expect("failed to get layer name str");

            if available_layers.iter().any(|prop| {
                unsafe { CStr::from_ptr(prop.layer_name.as_ptr()) }
                    .to_str()
                    .unwrap()
                    == str
            }) {
                println!("{}", str);
            } else {
                panic!("required layers were not available!");
            }
        }
    }
}

pub struct PhysicalDeviceSelector<'a> {
    instance: Arc<Instance>,
    surface: Arc<VulkanSurface>,
    criteria: SelectionCriteria<'a>,
}

pub struct SelectionCriteria<'a> {
    required_extensions: Vec<CString>,
    features: vk::PhysicalDeviceFeatures,
    features12: vk::PhysicalDeviceVulkan12Features<'a>,
    features13: vk::PhysicalDeviceVulkan13Features<'a>,
}

impl SelectionCriteria<'_> {
    pub fn new() -> Self {
        Self {
            required_extensions: vec![],
            features: vk::PhysicalDeviceFeatures::default(),
            features12: vk::PhysicalDeviceVulkan12Features::default(),
            features13: vk::PhysicalDeviceVulkan13Features::default(),
        }
    }
}

impl<'a> PhysicalDeviceSelector<'a> {
    pub fn new(instance: Arc<Instance>, surface: Arc<VulkanSurface>) -> PhysicalDeviceSelector<'a> {
        Self {
            instance,
            surface,
            criteria: SelectionCriteria::new(),
        }
    }

    pub fn set_required_extensions(mut self, required_extensions: Vec<CString>) -> Self {
        self.criteria.required_extensions = required_extensions;
        self
    }

    pub fn set_required_features(mut self, required_features: vk::PhysicalDeviceFeatures) -> Self {
        self.criteria.features = required_features;
        self
    }

    pub fn set_required_features_12(
        mut self,
        required_features: vk::PhysicalDeviceVulkan12Features<'a>,
    ) -> Self {
        self.criteria.features12 = required_features;
        self
    }

    pub fn set_required_features_13(
        mut self,
        required_features: vk::PhysicalDeviceVulkan13Features<'a>,
    ) -> Self {
        self.criteria.features13 = required_features;
        self
    }

    pub fn select(&self) -> Option<BootstrapPhysicalDevice> {
        let physical_devices = unsafe { self.instance.handle.enumerate_physical_devices() }
            .expect("failed to find physical devices!");

        if physical_devices.len() == 0 {
            panic!("failed to find GPUs with Vulkan support!");
        }

        let mut scored_devices = physical_devices
            .iter()
            .filter_map(|device| {
                if let Some((score, physical_device)) = self.rate_device(device) {
                    Some((score, physical_device))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        scored_devices.sort_by(|(a, _), (b, _)| a.cmp(b));

        if let Some((_, device)) = scored_devices.last() {
            Some(device.clone())
        } else {
            None
        }
    }

    fn rate_device(&self, device: &vk::PhysicalDevice) -> Option<(u32, BootstrapPhysicalDevice)> {
        let mut device_features_13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut device_features_12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut physical_device_features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut device_features_12)
            .push_next(&mut device_features_13);
        unsafe {
            self.instance
                .handle
                .get_physical_device_features2(*device, &mut physical_device_features)
        };
        let features = physical_device_features.features;

        let has_required_features = self.check_for_required_features(features)
            && self.check_for_required_features_12(device_features_12)
            && self.check_for_required_features_13(device_features_13);

        if !has_required_features {
            println!("failed to get required features!");
            return None;
        }

        let has_required_extensions = self.check_for_required_extensions(*device);

        if !has_required_extensions {
            println!("failed to get required extensions!");
            return None;
        }

        let mut score = 0;

        let properties = unsafe { self.instance.handle.get_physical_device_properties(*device) };

        if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            // Discrete GPUs have a significant performance advantage
            score += 1000;
        }

        // Maximum possible size of textures affects graphics quality
        score += properties.limits.max_image_dimension2_d;

        let physical_device_properties =
            unsafe { self.instance.handle.get_physical_device_properties(*device) };
        let device_name =
            unsafe { CStr::from_ptr(physical_device_properties.device_name.as_ptr()).to_str() }
                .unwrap();
        let queue_families = unsafe {
            self.instance
                .handle
                .get_physical_device_queue_family_properties(*device)
        };

        let queue_family_indices = QueueFamilyIndices::new(&queue_families, &self.surface, device);
        if queue_family_indices.graphics_family.is_some() {
            let swapchain_support_details = SwapchainSupportDetails::new(device, &self.surface);

            if swapchain_support_details.formats.len() > 0
                && swapchain_support_details.present_modes.len() > 0
            {
                let msaa_samples = Self::get_max_usable_sample_count(physical_device_properties);

                Some((
                    score,
                    BootstrapPhysicalDevice {
                        physical_device: *device,
                        physical_device_properties,
                        swapchain_support_details,
                        queue_family_indices,
                        max_sample_count: msaa_samples,
                    },
                ))
            } else {
                println!(
                    "failed to find swapchain format or present mode on physical device! {}",
                    device_name
                );
                None
            }
        } else {
            println!(
                "failed to find graphics family queue on physical device! {}",
                device_name
            );
            None
        }
    }

    pub fn get_max_usable_sample_count(
        physical_device_properties: vk::PhysicalDeviceProperties,
    ) -> vk::SampleCountFlags {
        let counts = physical_device_properties
            .limits
            .framebuffer_color_sample_counts
            & physical_device_properties
                .limits
                .framebuffer_depth_sample_counts;

        let sorted_desired_sample_counts = [
            vk::SampleCountFlags::TYPE_64,
            vk::SampleCountFlags::TYPE_32,
            vk::SampleCountFlags::TYPE_16,
            vk::SampleCountFlags::TYPE_8,
            vk::SampleCountFlags::TYPE_4,
            vk::SampleCountFlags::TYPE_2,
        ];

        let mut allowed_sample_counts = sorted_desired_sample_counts
            .iter()
            .filter(|desired_sample_count| counts.contains(**desired_sample_count));

        if let Some(sample_count) = allowed_sample_counts.next() {
            *sample_count
        } else {
            vk::SampleCountFlags::TYPE_1
        }
    }

    fn check_for_required_extensions(&self, device: vk::PhysicalDevice) -> bool {
        let device_extension_properties = unsafe {
            self.instance
                .handle
                .enumerate_device_extension_properties(device)
        }
        .expect("failed to get device extension properties!");
        let mut device_extension_names = device_extension_properties
            .iter()
            .map(|prop| unsafe { CString::from(CStr::from_ptr(prop.extension_name.as_ptr())) });
        for required_extension in &self.criteria.required_extensions {
            if device_extension_names
                .find(|extension| *extension == *required_extension)
                .is_none()
            {
                println!(
                    "couldn't find extension {}",
                    required_extension.to_str().unwrap()
                );
                return false;
            } else {
                println!("found extension {}", required_extension.to_str().unwrap());
            }
        }

        return true;
    }

    fn check_for_required_features(&self, device_features: vk::PhysicalDeviceFeatures) -> bool {
        let criteria_features = self.criteria.features;
        if criteria_features.geometry_shader == 1 && device_features.geometry_shader == 0 {
            false
        } else if criteria_features.sampler_anisotropy == 1
            && device_features.sampler_anisotropy == 0
        {
            false
        } else if criteria_features.alpha_to_one == 1 && device_features.alpha_to_one == 0 {
            false
        } else if criteria_features.depth_bias_clamp == 1 && device_features.depth_bias_clamp == 0 {
            false
        } else if criteria_features.depth_bounds == 1 && device_features.depth_bounds == 0 {
            false
        } else if criteria_features.depth_clamp == 1 && device_features.depth_clamp == 0 {
            false
        } else if criteria_features.draw_indirect_first_instance == 1
            && device_features.draw_indirect_first_instance == 0
        {
            false
        } else {
            true
        }
    }

    fn check_for_required_features_12(
        &self,
        device_features: vk::PhysicalDeviceVulkan12Features,
    ) -> bool {
        let criteria_features = self.criteria.features12;
        if criteria_features.buffer_device_address == 1
            && device_features.buffer_device_address == 0
        {
            false
        } else if criteria_features.descriptor_indexing == 1
            && device_features.descriptor_indexing == 0
        {
            false
        } else {
            true
        }
    }

    fn check_for_required_features_13(
        &self,
        device_features: vk::PhysicalDeviceVulkan13Features,
    ) -> bool {
        let criteria_features = self.criteria.features13;
        if criteria_features.dynamic_rendering == 1 && device_features.dynamic_rendering == 0 {
            false
        } else if criteria_features.synchronization2 == 1 && device_features.synchronization2 == 0 {
            false
        } else {
            true
        }
    }
}

#[derive(Clone)]
pub struct BootstrapPhysicalDevice {
    pub physical_device: vk::PhysicalDevice,
    pub queue_family_indices: QueueFamilyIndices,
    pub swapchain_support_details: SwapchainSupportDetails,
    pub max_sample_count: vk::SampleCountFlags,
    pub physical_device_properties: vk::PhysicalDeviceProperties,
}

pub fn create_logical_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_indices: &QueueFamilyIndices,
    required_extensions: Vec<CString>,
    device_features: vk::PhysicalDeviceFeatures,
    mut device_features12: vk::PhysicalDeviceVulkan12Features,
    mut device_features13: vk::PhysicalDeviceVulkan13Features,
) -> Result<ash::Device, vk::Result> {
    let queue_create_infos = [queue_family_indices.graphics_family.unwrap() as u32].map(|i| {
        vk::DeviceQueueCreateInfo::default()
            .queue_family_index(i)
            .queue_priorities(&[1.])
    });

    let device_extension_names_raw = required_extensions
        .iter()
        .map(|ext| ext.as_ptr())
        .collect::<Vec<_>>();
    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_features(&device_features)
        .enabled_extension_names(&device_extension_names_raw)
        .push_next(&mut device_features12)
        .push_next(&mut device_features13);

    unsafe { instance.create_device(physical_device, &device_create_info, None) }
}

pub struct LogicalDevice {
    instance: Arc<Instance>,
    pub handle: ash::Device,
}

impl LogicalDevice {
    pub fn new(
        instance: Arc<Instance>,
        physical_device: vk::PhysicalDevice,
        queue_family_indices: &QueueFamilyIndices,
        required_extensions: Vec<CString>,
        device_features: vk::PhysicalDeviceFeatures,
        mut device_features12: vk::PhysicalDeviceVulkan12Features,
        mut device_features13: vk::PhysicalDeviceVulkan13Features,
    ) -> std::result::Result<Arc<LogicalDevice>, ash::vk::Result> {
        let queue_create_infos = [queue_family_indices.graphics_family.unwrap() as u32].map(|i| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(i)
                .queue_priorities(&[1.])
        });

        let device_extension_names_raw = required_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&device_features)
            .enabled_extension_names(&device_extension_names_raw)
            .push_next(&mut device_features12)
            .push_next(&mut device_features13);

        let device_result = unsafe {
            instance
                .handle
                .create_device(physical_device, &device_create_info, None)
        };

        if let Ok(device) = device_result {
            Ok(Arc::new(Self {
                instance,
                handle: device,
            }))
        } else {
            Err(device_result.err().unwrap())
        }
    }
}

impl Drop for LogicalDevice {
    fn drop(&mut self) {
        unsafe { self.handle.destroy_device(None) };
    }
}

#[derive(Clone, Copy)]
pub struct QueueFamilyIndices {
    pub graphics_family: Option<usize>,
    pub present_family: Option<usize>,
}

impl QueueFamilyIndices {
    fn new(
        queue_families: &Vec<vk::QueueFamilyProperties>,
        surface: &VulkanSurface,
        pdevice: &vk::PhysicalDevice,
    ) -> Self {
        let graphics_family_index = if let Some((graphics_family_index, _)) =
            queue_families.iter().enumerate().find(|(_, queue)| {
                queue.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && queue.queue_flags.contains(vk::QueueFlags::COMPUTE)
            }) {
            Some(graphics_family_index)
        } else {
            None
        };

        let present_family_index = if let Some((present_family_index, _)) =
            queue_families.iter().enumerate().find(|(i, _)| {
                unsafe {
                    surface.loader.get_physical_device_surface_support(
                        *pdevice,
                        *i as u32,
                        surface.handle,
                    )
                }
                .unwrap()
            }) {
            Some(present_family_index)
        } else {
            None
        };

        Self {
            graphics_family: graphics_family_index,
            present_family: present_family_index,
        }
    }
}
