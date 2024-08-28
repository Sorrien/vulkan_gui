use std::{fs::File, io::BufReader};

use ash::vk::{self, ImageTiling, ImageUsageFlags, SampleCountFlags};
use ktx::KtxInfo;

use crate::{AllocatedImage, VulkanEngine};

pub fn load_ktx(vulkan_engine: &mut VulkanEngine) -> AllocatedImage {
    //let ktx_file = BufReader::new(File::open("assets/cubemap_yokohama_rgba.ktx").unwrap());
    let ktx_file = BufReader::new(File::open("assets/autumn_field_puresky_4K.ktx").unwrap());
    let ktx = ktx::Decoder::new(ktx_file).unwrap();
    let width = ktx.pixel_width().clamp(1, u32::MAX);
    let height = ktx.pixel_height().clamp(1, u32::MAX);
    let depth = ktx.pixel_depth().clamp(1, u32::MAX);
    //let mipmap_levels = ktx.mipmap_levels();
    ktx.gl_internal_format();

    let mut textures = ktx.read_textures();
    //let data = textures.map(|x| x).flatten().collect::<Vec<_>>();
    let data = textures.next().unwrap();
    println!("skybox texture data length: {}", data.len());

    vulkan_engine.create_allocated_texture_image_cubemap(
        &data,
        vk::Extent3D {
            width,
            height,
            depth,
        },
        vk::Format::R8G8B8A8_UNORM,
        ImageTiling::OPTIMAL,
        ImageUsageFlags::SAMPLED,
        gpu_allocator::MemoryLocation::GpuOnly,
        1,
        SampleCountFlags::TYPE_1,
    )
}
