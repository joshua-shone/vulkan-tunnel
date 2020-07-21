use vulkano::{
	instance::{Instance, PhysicalDevice},
	device::{Device, DeviceExtensions},
	image::{ImageUsage, SwapchainImage},
	swapchain,
	swapchain::{
		AcquireError, ColorSpace, FullscreenExclusive, PresentMode, SurfaceTransform, Swapchain,
		SwapchainCreationError,
	},
	buffer::{BufferUsage, CpuAccessibleBuffer},
	command_buffer::{AutoCommandBufferBuilder, DynamicState},
	pipeline::{
		GraphicsPipeline,
		viewport::{Viewport},
	},
	framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass},
	sync,
	sync::{FlushError, GpuFuture},
};

use vulkano_win::VkSurfaceBuild;

use winit::{
	event::{Event, WindowEvent},
	event_loop::{ControlFlow, EventLoop},
	window::{
		Window,
		WindowBuilder,
	}
};

use std::{
	process,
	sync::Arc,
	f32::consts,
	time::Instant,
};

use rand;

use cgmath;

#[derive(Default, Debug, Clone)]
struct Vertex {
	position: [f32; 3],
}
vulkano::impl_vertex!(Vertex, position);

fn main() {
	let required_extensions = vulkano_win::required_extensions();

	let instance = match Instance::new(None, &required_extensions, None) {
		Ok(i) => i,
		Err(err) => {
			eprintln!("Could not create Vulkan instance: {}", err);
			process::exit(1);
		}
	};

	let physical_device = match PhysicalDevice::enumerate(&instance).next() {
		Some(physical) => physical,
		None => {
			eprintln!("Could not find any Vulkan devices");
			process::exit(1);
		}
	};

	println!("Using Vulkan physical device '{}' type: {:?}", physical_device.name(), physical_device.ty());

	let event_loop = EventLoop::new();

	let window_builder = WindowBuilder::new()
		.with_title("Vulkan Tunnel")
		.with_inner_size(winit::dpi::LogicalSize::new(1024.0, 1024.0));

	let surface = match window_builder.build_vk_surface(&event_loop, instance.clone()) {
		Ok(s) => s,
		Err(err) => {
			eprintln!("Could not create Vulkan window surface: {}", err);
			process::exit(1);
		}
	};

	let queue_family = match physical_device.queue_families().find(|&q| {
		q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
	}) {
		Some(q) => q,
		None => {
			eprintln!("Could not find suitable Vulkan queue family");
			process::exit(1);
		}
	};

	let device_extensions = DeviceExtensions {
		khr_swapchain: true,
		..DeviceExtensions::none()
	};

	let (device, mut queues) = match Device::new(
		physical_device,
		physical_device.supported_features(),
		&device_extensions,
		[(queue_family, 0.5)].iter().cloned(),
	) {
		Ok((device, queues)) => (device, queues),
		Err(err) => {
			eprintln!("Could not initialize Vulkan device: {}", err);
			process::exit(1);
		}
	};

	let queue = queues.next().unwrap();

	let (mut swapchain, images) = {
		let caps = surface.capabilities(physical_device).unwrap();
		println!("Min image count: {}", caps.min_image_count);

		let alpha = caps.supported_composite_alpha.iter().next().unwrap();
		let format = caps.supported_formats[0].0;

		let dimensions: [u32; 2] = surface.window().inner_size().into();

		match Swapchain::new(
			device.clone(),
			surface.clone(),
			caps.min_image_count,
			format,
			dimensions,
			1,
			ImageUsage::color_attachment(),
			&queue,
			SurfaceTransform::Identity,
			alpha,
			PresentMode::Fifo,
			FullscreenExclusive::Default,
			true,
			ColorSpace::SrgbNonLinear,
		) {
			Ok((swapchain, images)) => (swapchain, images),
			Err(err) => {
				eprintln!("Could not create swapchain: {}", err);
				process::exit(1);
			}
		}
	};

	println!("Number of swapchain images: {}", images.len());

	let tunnel_depth = 20.0;
	let tunnel_depth_subdivs = 200;
	let tunnel_radial_subdivs = 200;
	let (vertices, indices) = create_tunnel_mesh(tunnel_depth, tunnel_depth_subdivs, tunnel_radial_subdivs);

	let vertex_buffer = {
		CpuAccessibleBuffer::from_iter(
			device.clone(),
			BufferUsage::all(),
			false,
			vertices.iter().cloned(),
		)
		.unwrap()
	};

	let index_buffer = {
		CpuAccessibleBuffer::from_iter(
			device.clone(),
			BufferUsage::all(),
			false,
			indices.iter().cloned(),
		)
		.unwrap()
	};

	mod vertex_shader_declaration {
		vulkano_shaders::shader! {
			ty: "vertex",
			src: "
				#version 450

				layout(location = 0) in vec3 position;

				layout(push_constant) uniform Camera {
					mat4 projection_matrix;
					float z;
				} camera;

				void main() {
					vec3 view_position = position;
					view_position.z += camera.z;
					gl_Position = camera.projection_matrix * vec4(view_position, 1.0);
				}
			"
		}
	}

	mod fragment_shader_declaration {
		vulkano_shaders::shader! {
			ty: "fragment",
			src: "
				#version 450

				layout(location = 0) out vec4 f_color;

				void main() {
					f_color = vec4(1.0, 0.0, 0.0, 1.0);
				}
			"
		}
	}

	let vertex_shader = match vertex_shader_declaration::Shader::load(device.clone()) {
		Ok(shader) => shader,
		Err(err) => {
			eprintln!("Could not load vertex shader: {}", err);
			process::exit(1);
		}
	};

	let fragment_shader = match fragment_shader_declaration::Shader::load(device.clone()) {
		Ok(shader) => shader,
		Err(err) => {
			eprintln!("Could not load fragment shader: {}", err);
			process::exit(1);
		}
	};

	let field_of_view_y = cgmath::Rad(45.0 * consts::PI / 180.0);
	let aspect = 1.0;

	let mut camera = vertex_shader_declaration::ty::Camera {
		z: -5.0,
		projection_matrix: cgmath::perspective(
			field_of_view_y,
			aspect,
			0.1, // Near
			100.0 // Far
		).into()
	};

	let render_pass = Arc::new(
		vulkano::single_pass_renderpass!(
			device.clone(),
			attachments: {
				color: {
					load: Clear,
					store: Store,
					format: swapchain.format(),
					samples: 1,
				}
			},
			pass: {
				color: [color],
				depth_stencil: {}
			}
		)
		.unwrap(),
	);

	let pipeline = match GraphicsPipeline::start()
			.vertex_input_single_buffer::<Vertex>()
			.vertex_shader(vertex_shader.main_entry_point(), ())
			.triangle_strip()
			.viewports_dynamic_scissors_irrelevant(1)
			.fragment_shader(fragment_shader.main_entry_point(), ())
			.render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
			.build(device.clone()) {
		Ok(p) => Arc::new(p),
		Err(err) => {
			eprintln!("Could not create graphics pipeline: {}", err);
			process::exit(1);
		}
	};

	let mut dynamic_state = DynamicState {
		line_width: None,
		viewports: None,
		scissors: None,
		compare_mask: None,
		write_mask: None,
		reference: None,
	};

	let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

	let mut recreate_swapchain = false;

	let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

	let start_timestamp = Instant::now();

	let tunnel_journey_duration_ms = 5000;

	event_loop.run(move |event, _, control_flow| {
		match event {
			Event::WindowEvent {event: WindowEvent::CloseRequested, ..} => *control_flow = ControlFlow::Exit,
			Event::WindowEvent {event: WindowEvent::Resized(_), ..} => recreate_swapchain = true,
			Event::RedrawEventsCleared => {
				previous_frame_end.as_mut().unwrap().cleanup_finished();

				if recreate_swapchain {
					let dimensions: [u32; 2] = surface.window().inner_size().into();
					let (new_swapchain, new_images) =
						match swapchain.recreate_with_dimensions(dimensions) {
							Ok(r) => r,
							Err(SwapchainCreationError::UnsupportedDimensions) => return,
							Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
						};

					swapchain = new_swapchain;
					framebuffers = window_size_dependent_setup(
						&new_images,
						render_pass.clone(),
						&mut dynamic_state,
					);
					recreate_swapchain = false;
				}

				let (image_num, suboptimal, acquire_future) =
					match swapchain::acquire_next_image(swapchain.clone(), None) {
						Ok(r) => r,
						Err(AcquireError::OutOfDate) => {
							recreate_swapchain = true;
							return;
						}
						Err(e) => panic!("Failed to acquire next image: {:?}", e),
					};

				if suboptimal {
					recreate_swapchain = true;
				}

				let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into()];

				let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
					device.clone(),
					queue.family(),
				)
				.unwrap();

				let journey_completion_ratio = start_timestamp.elapsed().as_millis() as f32 / tunnel_journey_duration_ms as f32;
				camera.z = -tunnel_depth * (1.0 - journey_completion_ratio);

				builder
					.begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
					.unwrap()
					.draw_indexed(
						pipeline.clone(),
						&dynamic_state,
						vertex_buffer.clone(),
						index_buffer.clone(),
						(),
						camera,
					)
					.unwrap()
					.end_render_pass()
					.unwrap();

				// Finish building the command buffer by calling `build`.
				let command_buffer = builder.build().unwrap();

				let future = previous_frame_end
					.take()
					.unwrap()
					.join(acquire_future)
					.then_execute(queue.clone(), command_buffer)
					.unwrap()
					.then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
					.then_signal_fence_and_flush();

				match future {
					Ok(future) => {
						previous_frame_end = Some(future.boxed());
					}
					Err(FlushError::OutOfDate) => {
						recreate_swapchain = true;
						previous_frame_end = Some(sync::now(device.clone()).boxed());
					}
					Err(e) => {
						println!("Failed to flush future: {:?}", e);
						previous_frame_end = Some(sync::now(device.clone()).boxed());
					}
				}
			}
			_ => (),
		}
	});
}

fn window_size_dependent_setup(
	images: &[Arc<SwapchainImage<Window>>],
	render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
	dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
	let dimensions = images[0].dimensions();

	let viewport = Viewport {
		origin: [0.0, 0.0],
		dimensions: [dimensions[0] as f32, dimensions[1] as f32],
		depth_range: 0.0..1.0,
	};
	dynamic_state.viewports = Some(vec![viewport]);

	images
		.iter()
		.map(|image| {
			Arc::new(
				Framebuffer::start(render_pass.clone())
					.add(image.clone())
					.unwrap()
					.build()
					.unwrap(),
			) as Arc<dyn FramebufferAbstract + Send + Sync>
		})
		.collect::<Vec<_>>()
}

fn create_tunnel_mesh(depth: f32, depth_subdivs: u32, radial_subdivs: u32) -> (Vec<Vertex>, Vec<u16>) {
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
