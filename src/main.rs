use glutin::ContextBuilder;
use winit::{
    event::{Event, WindowEvent},
    event_loop::ControlFlow,
    event_loop::EventLoop,
    window::WindowBuilder,
};
use std::{ffi::CString, mem, ptr, str};
use cgmath::{perspective, Deg, InnerSpace, Matrix, Matrix4, Point3, Vector3};
use rand::Rng;
use std::time::Instant;


fn compile_shader(src: &str, ty: gl::types::GLenum) -> Result<u32, String> {
    let shader;
    unsafe {
        shader = gl::CreateShader(ty);
        let c_str = CString::new(src.as_bytes()).unwrap();
        gl::ShaderSource(shader, 1, &c_str.as_ptr(), ptr::null());
        gl::CompileShader(shader);

        let mut success = gl::FALSE as gl::types::GLint;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);
        if success != gl::TRUE as i32 {
            let mut len = 0;
            gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
            let mut buffer = vec![0u8; len as usize];
            gl::GetShaderInfoLog(shader, len, ptr::null_mut(), buffer.as_mut_ptr() as *mut _);
            return Err(str::from_utf8(&buffer).unwrap().to_string());
        }
    }
    Ok(shader)
}

fn link_program(vs: u32, fs: u32) -> Result<u32, String> {
    let program;
    unsafe {
        program = gl::CreateProgram();
        gl::AttachShader(program, vs);
        gl::AttachShader(program, fs);
        gl::LinkProgram(program);

        let mut success = gl::FALSE as gl::types::GLint;
        gl::GetProgramiv(program, gl::LINK_STATUS, &mut success);
        if success != gl::TRUE as i32 {
            let mut len = 0;
            gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
            let mut buffer = vec![0u8; len as usize];
            gl::GetProgramInfoLog(program, len, ptr::null_mut(), buffer.as_mut_ptr() as *mut _);
            return Err(str::from_utf8(&buffer).unwrap().to_string());
        }
    }
    Ok(program)
}

fn main() {
    // Create event loop and window
    let event_loop = EventLoop::new();
    let window_builder = WindowBuilder::new()
        .with_title("OpenGL Lighting")
        .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0));

    let windowed_context = ContextBuilder::new()
        .with_vsync(true)
        .build_windowed(window_builder, &event_loop)
        .unwrap();

    let windowed_context = unsafe { windowed_context.make_current().unwrap() };

    // Load OpenGL functions
    gl::load_with(|symbol| windowed_context.get_proc_address(symbol) as *const _);

    // Set clear color
    unsafe {
        gl::ClearColor(0.0, 0.0, 0.0, 1.0);
    }

    // Triangle vertex data
    let vertices: [f32; 18] = [
        // positions         // normals
        -0.5, -0.5, 0.0,     0.0, 0.0, 1.0,
         0.5, -0.5, 0.0,     0.0, 0.0, 1.0,
         0.0,  0.5, 0.0,     0.0, 0.0, 1.0,
    ];

    // Vao stores the state related to vertex attribute pointers (position, color. normals, etc.)
    let (mut vao, mut vbo) = (0, 0);

    unsafe {
        // Generate a single vao id
        gl::GenVertexArrays(1, &mut vao);
        // Generate a single vbo
        gl::GenBuffers(1, &mut vbo);

        // Bind the vao to use/configure it
        gl::BindVertexArray(vao);

        // Bind the vbo and set the buffer data (positions & normals) associated with it
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);

        // At this point we don't distinguish between positions and normals in the data
        gl::BufferData(gl::ARRAY_BUFFER, 
            (vertices.len() * mem::size_of::<f32>()) as isize,
            vertices.as_ptr() as *const _, 
            gl::STATIC_DRAW);

        // Explain how to distinguish positions from normals in the data.  Note the attribute location
        // number (index) is referenced from the shader

        // Attribute location 0: position
        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, 6 * mem::size_of::<f32>() as i32, ptr::null());
        gl::EnableVertexAttribArray(0);

        // Attribute location 1: normals
        gl::VertexAttribPointer(1, 3, gl::FLOAT, gl::FALSE, 6 * mem::size_of::<f32>() as i32, (3 * mem::size_of::<f32>()) as *const _);
        gl::EnableVertexAttribArray(1);
    }

    // Shaders
    let vs_src = r#"
        #version 330 core
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aNormal;

        out vec3 FragPos;
        out vec3 Normal;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        void main() {
            FragPos = vec3(model * vec4(aPos, 1.0));
            Normal = mat3(transpose(inverse(model))) * aNormal;
            gl_Position = projection * view * vec4(FragPos, 1.0);
        }
    "#;

    let fs_src = r#"
        #version 330 core
        in vec3 FragPos;
        in vec3 Normal;
        out vec4 FragColor;

        uniform vec3 lightPos;
        uniform vec3 viewPos;
        uniform vec3 lightColor;
        uniform vec3 objectColor;

        void main() {
            float ambientStrength = 0.1;
            vec3 ambient = ambientStrength * lightColor;

            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;

            float specularStrength = 0.5;
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrength * spec * lightColor;

            vec3 result = (ambient + diffuse + specular) * objectColor;
            FragColor = vec4(result, 1.0);
        }
    "#;

    // Compile and link the shader programs
    let vs = compile_shader(vs_src, gl::VERTEX_SHADER).unwrap();
    let fs = compile_shader(fs_src, gl::FRAGMENT_SHADER).unwrap();
    let program = link_program(vs, fs).unwrap();

    // View matrix
    let view = Matrix4::look_at_rh(
        Point3::new(0.0, 0.0, 2.0),
        Point3::new(0.0, 0.0, 0.0),
        Vector3::unit_y(),
    );

    // Projection matrix
    let projection = perspective(Deg(45.0), 800.0 / 600.0, 0.1, 100.0);

    // Capture current time
    let start_time = Instant::now();

    // Choose a random axis of rotation
    let mut rng = rand::thread_rng();
    let rotation_axis = Vector3::new(
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
    ).normalize();

    // Define surface normals at each of the corners of the triangle
    let normal_scale = 0.1; // length of arrows
    let normal_lines: [f32; 18] = [
        // start point          // end point
        -0.5, -0.5, 0.0,        -0.5, -0.5, normal_scale,
         0.5, -0.5, 0.0,         0.5, -0.5, normal_scale,
         0.0,  0.5, 0.0,         0.0,  0.5, normal_scale,
    ];

    // Vao stores the state related to vertex attribute pointers (position, color. normals, etc.)
    let (mut arrow_vao, mut arrow_vbo) = (0, 0);

    unsafe {
        // Generate a single vao id
        gl::GenVertexArrays(1, &mut arrow_vao);

        // Generate a single vbo
        gl::GenBuffers(1, &mut arrow_vbo);
    
        // Bind the vao to use/configure it
        gl::BindVertexArray(arrow_vao);

        // Bind the vbo and set the buffer data (positions & normals) associated with it
        gl::BindBuffer(gl::ARRAY_BUFFER, arrow_vbo);

        // At this point we don't distinguish between positions and normals in the data
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (normal_lines.len() * mem::size_of::<f32>()) as isize,
            normal_lines.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );
    
        // Explain how to distinguish positions from normals in the data.  Note the attribute location
        // number (index) is referenced from the shader

        // Attribute location 0: start point & end point
        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, 3 * mem::size_of::<f32>() as i32, ptr::null());
        gl::EnableVertexAttribArray(0);
    }

    let arrow_vs = r#"
        #version 330 core
        layout(location = 0) in vec3 aPos;
        vec3 FragPos;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        void main() {
            FragPos = vec3(model * vec4(aPos, 1.0));
            gl_Position = projection * view * vec4(FragPos, 1.0);
        }
    "#;
    
    let arrow_fs = r#"
        #version 330 core
        out vec4 FragColor;
        void main() {
            FragColor = vec4(1.0, 1.0, 0.0, 1.0); // yellow
        }
    "#;

    // Compile and link the shader programs
    let arrow_vs_id = compile_shader(arrow_vs, gl::VERTEX_SHADER).unwrap();
    let arrow_fs_id = compile_shader(arrow_fs, gl::FRAGMENT_SHADER).unwrap();
    let arrow_program = link_program(arrow_vs_id, arrow_fs_id).unwrap();

    // Run event loop
    event_loop.run(move |event, _, control_flow| {

        *control_flow = ControlFlow::Poll;

        match event {
            Event::RedrawRequested(_) => {
                unsafe {
                    gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

                    // Use the main object shader (vertex and fragment) program
                    gl::UseProgram(program);

                    // Calculate rotation angle based on time passage
                    let elapsed = start_time.elapsed().as_secs_f32();
                    let angle = Deg(elapsed * 45.0); // 45 degrees per second

                    // Adjust the model matrix by rotating it around the randomly-chosen vector
                    let model = Matrix4::from_axis_angle(rotation_axis, angle);

                    // Gain access to the variables used in the shader program
                    let model_loc = gl::GetUniformLocation(program, CString::new("model").unwrap().as_ptr());
                    let view_loc = gl::GetUniformLocation(program, CString::new("view").unwrap().as_ptr());
                    let proj_loc = gl::GetUniformLocation(program, CString::new("projection").unwrap().as_ptr());
                    let light_pos_loc = gl::GetUniformLocation(program, CString::new("lightPos").unwrap().as_ptr());
                    let view_pos_loc = gl::GetUniformLocation(program, CString::new("viewPos").unwrap().as_ptr());
                    let light_color_loc = gl::GetUniformLocation(program, CString::new("lightColor").unwrap().as_ptr());
                    let object_color_loc = gl::GetUniformLocation(program, CString::new("objectColor").unwrap().as_ptr());

                    // Assign shader variable data
                    gl::UniformMatrix4fv(model_loc, 1, gl::FALSE, model.as_ptr());
                    gl::UniformMatrix4fv(view_loc, 1, gl::FALSE, view.as_ptr());
                    gl::UniformMatrix4fv(proj_loc, 1, gl::FALSE, projection.as_ptr());
                    gl::Uniform3f(light_pos_loc, 1.2, 1.0, 2.0);
                    gl::Uniform3f(view_pos_loc, 0.0, 0.0, 2.0);
                    gl::Uniform3f(light_color_loc, 1.0, 1.0, 1.0);
                    gl::Uniform3f(object_color_loc, 0.3, 0.5, 1.0);

                    // Select the triangle vao into context
                    gl::BindVertexArray(vao);

                    // Draw the triangle
                    gl::DrawArrays(gl::TRIANGLES, 0, 3);
        
                    // Use the normal vector shader (vertex and fragment) program
                    gl::UseProgram(arrow_program);
        
                    // Gain access to the variables used in the shader program
                    let model_loc = gl::GetUniformLocation(arrow_program, CString::new("model").unwrap().as_ptr());
                    let view_loc = gl::GetUniformLocation(arrow_program, CString::new("view").unwrap().as_ptr());
                    let proj_loc = gl::GetUniformLocation(arrow_program, CString::new("projection").unwrap().as_ptr());

                    // Assign shader variable data
                    gl::UniformMatrix4fv(model_loc, 1, gl::FALSE, model.as_ptr());
                    gl::UniformMatrix4fv(view_loc, 1, gl::FALSE, view.as_ptr());
                    gl::UniformMatrix4fv(proj_loc, 1, gl::FALSE, projection.as_ptr());
        
                    // Select the normal vectors vao into context
                    gl::BindVertexArray(arrow_vao);

                    // Draw the normal vectors as lines
                    gl::DrawArrays(gl::LINES, 0, 6); // 3 lines → 6 vertices
                }

                // Swap the backbuffer to display
                windowed_context.swap_buffers().unwrap();
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                _ => {}
            },
            Event::MainEventsCleared => {
                windowed_context.window().request_redraw();
            }
            _ => (),
        }
    });
}
