use std::{fs::File, io::Read, path::Path, sync::Arc};

use crate::{
    buffers::{GeoSurface, MeshAsset, Vertex},
    MaterialInstance, VulkanEngine,
};
use base64::prelude::*;

pub fn load_gltf_meshes<P>(
    engine: &mut VulkanEngine,
    path: P,
    default_material: Arc<MaterialInstance>,
) -> Option<Vec<MeshAsset>>
where
    P: AsRef<Path>,
{
    if let Ok(gltf) = gltf::Gltf::open(path) {
        let mut buffer_data = vec![];

        for buffer in gltf.buffers() {
            match buffer.source() {
                gltf::buffer::Source::Bin => {
                    if let Some(blob) = gltf.blob.as_deref() {
                        buffer_data.push(blob.into());
                        println!("Found a bin, saving");
                    };
                }
                gltf::buffer::Source::Uri(uri) => {
                    if let Ok(mut bin_file) = File::open(uri) {
                        let mut buf = vec![];
                        bin_file.read_to_end(&mut buf).unwrap();
                        buffer_data.push(buf);
                    } else {
                        let mut split = uri.split(",");
                        let metadata = split.next().unwrap();
                        let data = split.next().unwrap();
                        let mut metadata_split = metadata[5..].split(";");
                        let media_type = metadata_split.next().unwrap();
                        let encoding = metadata_split.next().unwrap();

                        if media_type == "application/octet-stream" && encoding == "base64" {
                            let decoded_data = BASE64_STANDARD.decode(data).unwrap();
                            buffer_data.push(decoded_data);
                        } else {
                            println!("media type: {} encoding: {}", media_type, encoding);
                            panic!("unknown media or encoding type!");
                        }
                    }
                }
            }
        }

        let mut mesh_assets = vec![];

        for scene in gltf.scenes() {
            for node in scene.nodes() {
                if let Some(mesh) = node.mesh() {
                    let primitives = mesh.primitives();

                    let mut indices = vec![];
                    let mut vertices = vec![];

                    let mut surfaces = vec![];

                    primitives.for_each(|primitive| {
                        let reader = primitive.reader(|buffer| Some(&buffer_data[buffer.index()]));

                        let initial_vertex = vertices.len();

                        let mut primitive_indices = vec![];
                        if let Some(indices_raw) = reader.read_indices() {
                            primitive_indices
                                .append(&mut indices_raw.into_u32().collect::<Vec<u32>>());
                        }

                        let new_surface = GeoSurface {
                            start_index: indices.len(),
                            count: primitive_indices.len(),
                            material: default_material.clone(),
                        };

                        surfaces.push(new_surface);

                        primitive_indices.iter().for_each(|index| {
                            indices.push(index + initial_vertex as u32);
                        });

                        let mut primitive_vertices = vec![];

                        if let Some(vertices_raw) = reader.read_positions() {
                            vertices_raw.for_each(|vertex| {
                                primitive_vertices
                                    .push(Vertex::new(glam::Vec3::from(vertex), glam::Vec4::ONE));
                            });
                        }

                        if let Some(normal_attribute) = reader.read_normals() {
                            normal_attribute.enumerate().for_each(|(i, normal)| {
                                primitive_vertices[i].normal = glam::Vec3::from(normal);
                            });
                        }

                        if let Some(tex_coord_attribute) =
                            reader.read_tex_coords(0).map(|v| v.into_f32())
                        {
                            tex_coord_attribute.enumerate().for_each(|(i, tex_coord)| {
                                primitive_vertices[i].uv_x = tex_coord[0];
                                primitive_vertices[i].uv_y = tex_coord[1];
                            });
                        }

                        if let Some(color_attribute) =
                            reader.read_colors(0).map(|c| c.into_rgba_f32())
                        {
                            color_attribute.enumerate().for_each(|(i, color)| {
                                primitive_vertices[i].color = glam::Vec4::from(color);
                            });
                        }

                        vertices.append(&mut primitive_vertices);
                    });

                    let override_colors = false;

                    if override_colors {
                        vertices.iter_mut().for_each(|vertex| {
                            vertex.color = glam::Vec4::new(
                                vertex.normal.x,
                                vertex.normal.y,
                                vertex.normal.z,
                                1.,
                            );
                        })
                    }
                    let mesh_buffers = engine.upload_mesh_gpu_only(indices, vertices);

                    let mesh_name = if let Some(name) = mesh.name() {
                        name
                    } else {
                        "default mesh name"
                    };

                    if mesh.name().unwrap() == "cube" {
                        println!("cube");
                    }

                    mesh_assets.push(MeshAsset {
                        name: mesh_name.to_string(),
                        surfaces,
                        mesh_buffers: Arc::new(mesh_buffers),
                    });
                }
            }
        }
        Some(mesh_assets)
    } else {
        None
    }
}
