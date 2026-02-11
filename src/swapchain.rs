use std::sync::Arc;

use ash::{khr::swapchain::Device as Swapchain, vk};

use ash_bootstrap::{
    swapchain::SwapchainSupportDetails, Instance, LogicalDevice, QueueFamilyIndices, VulkanSurface,
};

pub struct MySwapchain {
    device: Arc<LogicalDevice>,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_loader: Swapchain,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    is_valid: bool,
}

impl MySwapchain {
    pub fn builder(
        instance: Arc<Instance>,
        device: Arc<LogicalDevice>,
        surface: Arc<VulkanSurface>,
        swapchain_support: SwapchainSupportDetails,
        queue_family_indices: QueueFamilyIndices,
    ) -> SwapchainBuilder {
        SwapchainBuilder::new(
            instance,
            device,
            surface,
            swapchain_support,
            queue_family_indices,
        )
    }

    pub fn destroy(&mut self) {
        self.is_valid = false;
        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None)
        };

        for i in 0..self.swapchain_image_views.len() {
            let swapchain_image_view = self.swapchain_image_views[i];
            unsafe {
                self.device
                    .handle
                    .destroy_image_view(swapchain_image_view, None)
            };
        }
    }
}

impl Drop for MySwapchain {
    fn drop(&mut self) {
        if self.is_valid {
            self.destroy();
        }
    }
}

pub struct SwapchainBuilder {
    swapchain_support: SwapchainSupportDetails,
    queue_family_indices: QueueFamilyIndices,
    instance: Arc<Instance>,
    device: Arc<LogicalDevice>,
    surface: Arc<VulkanSurface>,
    window_width: u32,
    window_height: u32,
    desired_present_mode: vk::PresentModeKHR,
    desired_surface_format: vk::SurfaceFormatKHR,
    swapchain_image_usage_flags: vk::ImageUsageFlags,
}

impl SwapchainBuilder {
    pub fn new(
        instance: Arc<Instance>,
        device: Arc<LogicalDevice>,
        surface: Arc<VulkanSurface>,
        swapchain_support: SwapchainSupportDetails,
        queue_family_indices: QueueFamilyIndices,
    ) -> Self {
        Self {
            swapchain_support,
            queue_family_indices,
            instance,
            device,
            surface,
            window_width: 800,
            window_height: 600,
            desired_present_mode: vk::PresentModeKHR::MAILBOX,
            desired_surface_format: vk::SurfaceFormatKHR::default()
                .format(vk::Format::B8G8R8A8_SRGB)
                .color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR),
            swapchain_image_usage_flags: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        }
    }

    pub fn desired_extent(mut self, window_width: u32, window_height: u32) -> Self {
        self.window_height = window_height;
        self.window_width = window_width;
        self
    }

    pub fn desired_present_mode(mut self, desired_present_mode: vk::PresentModeKHR) -> Self {
        self.desired_present_mode = desired_present_mode;
        self
    }

    pub fn desired_surface_format(mut self, desired_surface_format: vk::SurfaceFormatKHR) -> Self {
        self.desired_surface_format = desired_surface_format;
        self
    }

    pub fn add_image_usage_flags(mut self, image_usage_flags: vk::ImageUsageFlags) -> Self {
        self.swapchain_image_usage_flags |= image_usage_flags;
        self
    }

    pub fn build(self) -> MySwapchain {
        let (swapchain, swapchain_loader, format, extent) = self
            .create_swapchain(self.window_width, self.window_height)
            .expect("failed to create swapchain!");
        let swapchain_images = Self::create_swapchain_images(&swapchain, &swapchain_loader)
            .expect("failed to get swapchain images!");
        let swapchain_image_views = self.create_swapchain_image_views(&swapchain_images, format);

        MySwapchain {
            device: self.device,
            swapchain,
            swapchain_loader,
            format,
            extent,
            swapchain_images,
            swapchain_image_views,
            is_valid: true,
        }
    }
    fn create_swapchain(
        &self,
        window_width: u32,
        window_height: u32,
    ) -> Result<(vk::SwapchainKHR, Swapchain, vk::Format, vk::Extent2D), vk::Result> {
        let surface_format = self
            .choose_swapchain_surface_format(&self.swapchain_support.formats)
            .expect("failed to find surface format!");
        let present_mode =
            self.choose_swapchain_present_mode(&self.swapchain_support.present_modes);
        let extent = Self::choose_swap_extent(
            &self.swapchain_support.capabilities,
            window_width,
            window_height,
        );

        let image_count = self.swapchain_support.capabilities.min_image_count + 1;

        //if max_image_count is 0 then there is no max
        let image_count = if self.swapchain_support.capabilities.max_image_count > 0
            && image_count > self.swapchain_support.capabilities.max_image_count
        {
            self.swapchain_support.capabilities.max_image_count
        } else {
            image_count
        };

        let (sharing_mode, queue_indices) = if self.queue_family_indices.graphics_family
            != self.queue_family_indices.present_family
        {
            (
                vk::SharingMode::CONCURRENT,
                vec![
                    self.queue_family_indices.graphics_family.unwrap() as u32,
                    self.queue_family_indices.present_family.unwrap() as u32,
                ],
            )
        } else {
            (vk::SharingMode::EXCLUSIVE, vec![])
        };

        let swapchain_loader = Swapchain::new(&self.instance.handle, &self.device.handle);
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface.handle)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(self.swapchain_image_usage_flags)
            .image_sharing_mode(sharing_mode)
            .queue_family_indices(&queue_indices)
            .pre_transform(self.swapchain_support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None) }?;

        Ok((swapchain, swapchain_loader, surface_format.format, extent))
    }

    fn choose_swapchain_surface_format<'a>(
        &'a self,
        available_formats: &'a Vec<vk::SurfaceFormatKHR>,
    ) -> Option<&'a vk::SurfaceFormatKHR> {
        if let Some(desired_format) = available_formats.iter().find(|surface_format| {
            surface_format.color_space == self.desired_surface_format.color_space
                && surface_format.format == self.desired_surface_format.format
        }) {
            Some(desired_format)
        } else {
            available_formats.first()
        }
    }

    fn choose_swapchain_present_mode(
        &self,
        available_present_modes: &Vec<vk::PresentModeKHR>,
    ) -> vk::PresentModeKHR {
        let desired_mode = self.desired_present_mode;
        let is_desired_mode_available = available_present_modes
            .iter()
            .any(|present_mode| *present_mode == desired_mode);
        if is_desired_mode_available {
            desired_mode
        } else {
            vk::PresentModeKHR::FIFO
        }
    }

    fn choose_swap_extent(
        capabilities: &vk::SurfaceCapabilitiesKHR,
        window_width: u32,
        window_height: u32,
    ) -> vk::Extent2D {
        match capabilities.current_extent.width {
            //the max value of u32 is a special value to indicate that we must choose a resolution with the current min and max extents
            //should look into how DPI scaling is handled by winit and if this is the pixel extent or if this includes dpi scaling.
            u32::MAX => vk::Extent2D {
                width: window_width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: window_height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            },
            _ => capabilities.current_extent,
        }
    }

    fn create_swapchain_images(
        swapchain: &vk::SwapchainKHR,
        swapchain_loader: &Swapchain,
    ) -> Result<Vec<vk::Image>, vk::Result> {
        unsafe { swapchain_loader.get_swapchain_images(*swapchain) }
    }

    fn create_swapchain_image_views(
        &self,
        swapchain_images: &Vec<vk::Image>,
        swapchain_image_format: vk::Format,
    ) -> Vec<vk::ImageView> {
        swapchain_images
            .iter()
            .map(|image| {
                self.create_image_view(
                    *image,
                    swapchain_image_format,
                    vk::ImageAspectFlags::COLOR,
                    1,
                )
            })
            .collect::<Vec<_>>()
    }

    fn create_image_view(
        &self,
        image: vk::Image,
        format: vk::Format,
        aspect_mask: vk::ImageAspectFlags,
        mip_levels: u32,
    ) -> vk::ImageView {
        let component_mapping = vk::ComponentMapping::default();
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect_mask)
            .base_mip_level(0)
            .level_count(mip_levels)
            .base_array_layer(0)
            .layer_count(1);

        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
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
}
