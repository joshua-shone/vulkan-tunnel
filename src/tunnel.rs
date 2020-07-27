use std::{
  process,
  f32::consts,
};
use rand;

use vulkano::{
	pipeline::{
		GraphicsPipeline,
	},
	framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass},
};

use crate::vertex::Vertex;

pub mod vertex_shader_decl {
	vulkano_shaders::shader! {
		ty: "vertex",
		src: "
			#version 450

			layout(location = 0) in vec3 position;

			layout(push_constant) uniform Camera {
				mat4 projection_matrix;
				float z;
			} camera;

			layout(location = 0) out vec3 view_position;

			void main() {
				view_position = position;
				view_position.z += camera.z;
				gl_Position = camera.projection_matrix * vec4(view_position, 1.0);
			}
		"
	}
}

pub mod fragment_shader_decl {
	vulkano_shaders::shader! {
		ty: "fragment",
		src: "
			#version 450

			layout(location = 0) in vec3 view_position;

			layout(location = 0) out vec4 f_color;

			const float tunnel_depth = 20.0;
			const vec3 diffuse = vec3(.3, .2, .03);
			const vec3 light_direction = vec3(0, 0, -1);

			void main() {
				vec3 normal = normalize(cross(dFdx(view_position), dFdy(view_position)));
				float incidence = max(dot(normal, light_direction), 0.0);
				float tunnel_end_closeness = 1.0 - (view_position.z / tunnel_depth);
				f_color.a = 1.0;
				f_color.rgb = diffuse * tunnel_end_closeness * (0.5 + incidence);
			}
		"
	}
}

fn create_pipeline() {
	let vertex_shader = match vertex_shader_decl::Shader::load(device.clone()) {
		Ok(shader) => shader,
		Err(err) => {
			eprintln!("Could not load vertex shader: {}", err);
			process::exit(1);
		}
	};

	let fragment_shader = match fragment_shader_decl::Shader::load(device.clone()) {
		Ok(shader) => shader,
		Err(err) => {
			eprintln!("Could not load fragment shader: {}", err);
			process::exit(1);
		}
	};

	GraphicsPipeline::start()
	.vertex_input_single_buffer::<Vertex>()
	.vertex_shader(vertex_shader.main_entry_point(), ())
	.triangle_strip()
	.viewports_dynamic_scissors_irrelevant(1)
	.fragment_shader(fragment_shader.main_entry_point(), ())
	.render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
	.build(device.clone());
}

fn create_mesh(depth: f32, depth_subdivs: u32, radial_subdivs: u32) -> (Vec<Vertex>, Vec<u16>) {
	let mut vertices = Vec::<Vertex>::with_capacity((depth_subdivs * radial_subdivs * 3) as usize);
	let mut indices = Vec::<u16>::with_capacity(((depth_subdivs-1) * 6 * (radial_subdivs+1)) as usize);

	fn noise() -> f32 {
		return (rand::random::<f32>() - 0.5) * 0.1;
	}

	let slice_angle = (consts::PI * 2.0) / radial_subdivs as f32;

	for j in 0..depth_subdivs {
		for i in 0..radial_subdivs {
			let angle = slice_angle * i as f32;
			vertices.push(Vertex { position: [
				angle.cos() + noise(),
				angle.sin() + noise(),
				(j as f32 * (depth / depth_subdivs as f32)) + noise(),
			]});
		}
	}

	for j in 0..depth_subdivs {
		for i in 0..radial_subdivs {
			let mi  =  i    % radial_subdivs;
			let mi2 = (i+1) % radial_subdivs;
			indices.push(((j+1) * radial_subdivs + mi)  as u16);
			indices.push(( j    * radial_subdivs + mi)  as u16); // mesh[j][mi]
			indices.push(((j)   * radial_subdivs + mi2) as u16);
			indices.push(((j+1) * radial_subdivs + mi)  as u16);
			indices.push(( j    * radial_subdivs + mi2) as u16);
			indices.push(((j+1) * radial_subdivs + mi2) as u16);
		}
	}

	return (vertices, indices);
}
