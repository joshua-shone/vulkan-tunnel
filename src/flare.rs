pub mod vertex_shader_decl {
	vulkano_shaders::shader! {
		ty: "vertex",
		src: "
			#version 450
			layout(location = 0) in vec4 vertex_position;
			void main() { gl_Position = vertex_position; }
		"
	}
}

pub mod fragment_shader_decl {
	vulkano_shaders::shader! {
		ty: "fragment",
		src: "
			#version 450

			precision mediump float;

			layout(location = 0) out vec4 f_color;

			layout(push_constant) uniform Params {
				vec2 resolution;
				float timeMs;
			} params;

			void main() {
				vec2 uv = (gl_FragCoord.xy / params.resolution) - vec2(.5,.5);
				const float PI = 3.14159;
				float angle = 1.0 + (atan(uv.x, uv.y) / PI);
				float warble1 = sin(angle * 16.0);
				float warble2 = cos(angle * 20.0);
				float warble3 = sin((angle + (params.timeMs / 3000.0)) * 35.0);
				float warbles = (warble1 + (warble2*.5) + (warble3*.2)) / 12.0;
				float dist = 1.0 - length(uv);
				float fadein = min(params.timeMs / 1000.0, 1.0);
				float fadeout = 1.0 - (params.timeMs / 5000.0);
				float intensity = (pow(dist, 10.0) + warbles) * (fadeout * fadein);
				f_color = vec4(intensity);
			}
		"
	}
}
