use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
};

use shaderc::{self, IncludeCallbackResult, IncludeType, ResolvedInclude};

static mut FILE_PATHS: Vec<PathBuf> = Vec::<PathBuf>::new();

fn main() {
    unsafe { FILE_PATHS = get_all_files("shaders") };

    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.add_macro_definition("EP", Some("main"));
    options.set_include_callback(include_shader);

    for path in unsafe { FILE_PATHS.clone() } {
        let extension = path.extension().unwrap();
        if extension != "spv" && extension != "glsl" {
            let shader_kind = match extension.to_str().unwrap() {
                "frag" => shaderc::ShaderKind::Fragment,
                "vert" => shaderc::ShaderKind::Vertex,
                "comp" => shaderc::ShaderKind::Compute,
                "geom" => shaderc::ShaderKind::Geometry,
                "mesh" => shaderc::ShaderKind::Mesh,
                "rgen" => shaderc::ShaderKind::RayGeneration,
                "tesc" => shaderc::ShaderKind::TessControl,
                "tese" => shaderc::ShaderKind::TessEvaluation,
                "task" => shaderc::ShaderKind::Task,
                "rint" => shaderc::ShaderKind::Intersection,
                "rahit" => shaderc::ShaderKind::AnyHit,
                "rchit" => shaderc::ShaderKind::ClosestHit,
                "rmiss" => shaderc::ShaderKind::Miss,
                "rcall" => shaderc::ShaderKind::Callable,
                _ => shaderc::ShaderKind::InferFromSource,
            };

            let source = load_source(path.clone());
            println!("current path: {:?}", path);

            let binary_result = compiler
                .compile_into_spirv(
                    &source,
                    shader_kind,
                    path.file_name().unwrap().to_str().unwrap(),
                    "main",
                    Some(&options),
                )
                .unwrap();

            let mut new_path = String::from(path.to_str().unwrap());

            new_path.push_str(".spv");
            println!("new path: {:?}", new_path);
            let mut new_file = std::fs::File::create(new_path).unwrap();
            new_file.write_all(binary_result.as_binary_u8()).unwrap();
        }
    }
}

fn load_source<P>(path: P) -> String
where
    P: AsRef<Path>,
{
    let mut source_file = std::fs::File::open(path).unwrap();
    let mut source = String::new();
    source_file.read_to_string(&mut source).unwrap();

    source
}

fn get_all_files<P>(dir: P) -> Vec<PathBuf>
where
    P: AsRef<Path>,
{
    let mut file_paths = vec![];
    for path in std::fs::read_dir(dir).unwrap() {
        let path = path.unwrap().path();

        if path.is_file() {
            file_paths.push(path);
        } else {
            file_paths.append(&mut get_all_files(path));
        }
    }
    file_paths
}

fn include_shader(
    requested_source: &str,
    _include_type: IncludeType,
    _requesting_source: &str,
    _include_depth: usize,
) -> IncludeCallbackResult {
    unsafe { FILE_PATHS.clone() }.iter().find(|path| {
        let string = String::from(path.file_name().unwrap().to_str().unwrap());
        string.contains(requested_source)
    });
    let path = format!("shaders/{}", requested_source);
    let content = load_source(path.clone());
    IncludeCallbackResult::Ok(ResolvedInclude {
        resolved_name: path,
        content,
    })
}
