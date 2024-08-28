#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "vertex_definition.glsl"

layout(set = 0, binding = 0) uniform SceneData {

    mat4 view;
    mat4 proj;
    mat4 viewproj;
    vec4 ambientColor;
    vec4 sunlightDirection; //w for sun power
    vec4 sunlightColor;
} sceneData;

layout(location = 0) out vec3 outUVW;

layout(buffer_reference, std430) readonly buffer VertexBuffer {
Vertex vertices[];
};

//push constants block
layout(push_constant) uniform constants {
mat4 render_matrix;
VertexBuffer vertexBuffer;
}
PushConstants;

void main() {
Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];
outUVW = v.position;

//mat4 viewMat = mat4(mat3(PushConstants.render_matrix));
//gl_Position = sceneData.proj * viewMat * vec4(v.position.xyz, 1.0);

mat4 viewMat = mat4(mat3(sceneData.view));
gl_Position = sceneData.proj * viewMat * vec4(v.position.xyz, 1.0);
}