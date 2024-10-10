use anyhow::Result;
use ash::vk;
use presser::Slab;

/// A simple struct that wraps our Vulkan function pointer tables and application state
struct Context {
    #[allow(unused)]
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    physical_device: vk::PhysicalDevice,
    #[allow(unused)]
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
}

impl Context {
    /// Creates a new [`Context`] with reasonable defaults.
    unsafe fn new() -> Result<Self> {
        // An "entry" is an ash (the Rust Vulkan wrapper we're using) concept. Morally speaking it
        // populates a function-pointer table with functions supplied by your Vulkan driver.
        //
        // It's actually a pretty neat system and the doccumentation is quite readable
        // https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderInterfaceArchitecture.md
        //
        // `load` is the "just make it work please" magic function, and we'll do just that.
        let entry = ash::Entry::load()?;

        // Next up we create an "instance". This is our first *actual* Vulkan "handle".
        // // Vulkan's main schtick (especially in contrast to OpenGL) is that there is no global
        // state - everything is explicitly scoped. An "instance" is the top level per-application
        // state object that the driver uses to track state. You can have multiple instances in an
        // application, backed by separate drivers and everything will work fine. In practice, of
        // course, nobody does that and you just create a single instance.
        //
        // A handle is exactly what you'd expect; an opaque handle to some bit of driver state.
        // This has the neat property of making things *really* easy in Rust; since they're just
        // newtypes around u64, they can be trivially copied!
        //
        // OK. So how do we actually *create* an instance?
        //
        // The main pattern you'll see in ash is create_a_thing(&vk::ThingCreateInfo::default().thing(0)),
        // which is just a Rusty, more ergonomic version of
        //
        // ```cpp
        // vkCreateThing(VkThingCreateInfo {
        //  thing: 0,
        // })
        // ```
        //
        // But how does one know what functions to call and what parameters should be passed to it?
        //
        // Most of the time we're going to be calling "device commands", which are the actual meat
        // and potatoes of making the GPU work. But in this bootstrapping period we call functions
        // on `Entry` to get an `Instance`, and then a function on `Instance` to get a `Device`.
        //
        // In this instance we can take a look at the Vulkan spec and see that to create an
        // instance we must call [vkCreateInstance](https://registry.khronos.org/vulkan/specs/1.3/html/chap4.html#vkCreateInstance),
        // and that this function takes a structure of type [VkInstanceCreateInfo](https://registry.khronos.org/vulkan/specs/1.3/html/chap4.html#VkInstanceCreateInfo).
        //
        // Simple enough:
        let instance = entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(
                &vk::ApplicationInfo::default()
                    // We want to use Vulkan 1.3 as that's the latest
                    .api_version(vk::API_VERSION_1_3)
                    // `c""` is a quick little way of creating a &CStr
                    .application_name(c"Hello Compute"),
            ),
            // The last parameter to a createXXX function is typically an "alllocation callback",
            // which I suppose is useful if you're doing C shennanigans; we are not.
            None,
        )?;

        // Note that ash takes care of a *lot* of the bullshit for us. Since we don't really care
        // about most of the members of that struct, we can just defer to the default values.
        //
        // Also note that we populated the "struct builder" inline; this is very intentional.
        // As you can see, VkInstanceCreateInfo contains a pointer-to a VkApplicationInfo, and
        // you know how much Rust *loves* pointer shennanigans.
        //
        // To work around this, ash does some clever borrowchecker shennanigans to make sure all
        // the pointers work at they should and nothing gets dropped before the function is called;
        // for best results though it's *much* easier to just declare everything inline and let the
        // compiler be satisfied that no evil aliasing is occurring.

        // Okay, moving on. Let's get a `Device` to start calling commands on.
        //
        // A `Device` is a "logical device"; that is, it can be used to represent one or more
        // physical devices on your machine. I've never actually played around with that feature
        // much but I imagine it might be important in a HPC setup!
        //
        // First we need to find a physical device that meets our needs. On my PC, I've just got
        // the single NVIDIA card but laptops often have a useless integrated GPU that we probably
        // want to ignore. However, here I've just picked the first physical device that supports
        // compute commands.
        //
        // We query the device's compute support by iterating through its "queues". Unlike OpenGL
        // before it, Vulkan makes no attempt to hide the asynchrony between CPU and GPU and
        // presents an interface where work is batched up in command buffers and submitted to
        // a queue for execution.
        //
        // There's some great documentation on this [in the fundamentals section of the spec]
        // (https://registry.khronos.org/vulkan/specs/1.3/html/chap3.html#fundamentals-execmodel).
        //
        // The logic here is simple; if you have a queue family that supports compute, great, you
        // support compute. We also take note of this queue family's index, as we'll need it later.
        let (physical_device, compute_queue_family_index) = instance
            .enumerate_physical_devices()?
            .into_iter()
            .find_map(|physical_device| {
                // `find_map` is one of my favourite iterator patterns in Rust; it iterates through
                // the iterator and returns the first time the provided closure returns `Some` with
                // whatever data you like. It's very cute.

                // First, get the queue families for this device:
                let queue_families =
                    instance.get_physical_device_queue_family_properties(physical_device);

                // Now, iterate through these families with `enumerate`, which will also give us
                // the index in the vec. Note we use `into_iter` on both these collections since
                // we're happy to just drop the collection on the ground.
                let compute_queue_family_index =
                    queue_families
                        .into_iter()
                        .enumerate()
                        .find_map(|(index, queue_family)| {
                            // You could be cute here and do contains compute flags but NOT
                            // graphics to find a "compute only" queue but that's probably
                            // a waste of time.
                            if queue_family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                                Some(index as u32)
                            } else {
                                None
                            }
                        })?;

                Some((physical_device, compute_queue_family_index))
            })
            // I'm genuinely curious if there's any Vulkan device that doesn't support compute
            .ok_or_else(|| {
                anyhow::anyhow!("Unable to find a physical device with compute support")
            })?;

        // Great. Now that we have our physical device and the index of our compute queue, we can
        // create a logical device.
        //
        // Note that the return type is `ash::Device`, not `ash::vk::Device`; that's because
        // ash will populate a new device specific function pointer table that we want to carry
        // around for when we actually want to execute commands. This is the last time we'll
        // have to deal with that kind of stuff.
        //
        // When we create a device we also create one or more queues on that device, so we must
        // supply configuration to describe them. Note that we are again doing all of this inline
        // for simplicity.
        //
        // One fun quirk is that the `default` implementation will set `queue_prorities` to an
        // empty slice, so we need to pass it a single value to avoid Vulkan complaining that
        // we didn't specify the priority for the queue.
        //
        // Lastly, note the use of "push_next" here; this is a fun trick that Vulkan uses to extend
        // structures in newer versions or extensions:
        // https://registry.khronos.org/vulkan/specs/1.3/html/chap3.html#fundamentals-validusage-pNext
        //
        // More docs on devices and queues can be found here:
        // https://registry.khronos.org/vulkan/specs/1.3/html/chap5.html#devsandqueues-device-creation
        let device = instance.create_device(
            physical_device,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(&[vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(compute_queue_family_index)
                    .queue_priorities(&[1.0])])
                .push_next(
                    // We'll be using Buffer Device Address to avoid the tedium of descriptor sets
                    &mut vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true),
                ),
            None,
        )?;

        // Now that we've got all that done, let's fetch the queue that was created for us. The
        // second parameter here is the queue index, which in this case is 0 since we only created
        // a single queue.
        //
        // https://registry.khronos.org/vulkan/specs/1.3/html/chap5.html#vkGetDeviceQueue
        let queue = device.get_device_queue(compute_queue_family_index, 0);

        // Finally, we'll need a command buffer to actually record commands to the queue. We do
        // this by first creating a "command pool"...
        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(compute_queue_family_index),
                None,
            )?
        };

        // ...and then allocating ourselves a command buffer from that pool.
        let command_buffer = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_buffer_count(1)
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY),
            )
        }?[0];

        // And we're done! Before we finish let's retrieve the name of the device (as provided
        // by the driver) and log it to the terminal.
        let physical_device_properties = instance.get_physical_device_properties(physical_device);

        // Now, some people will say "what's with all the unchecked and transmute blah blah blah",
        // but the fun thing about Vulkan is that we can *guarantee* that our driver is conformant,
        // the spec is very strict about null pointers, array sizes etc. This makes life much
        // easier.
        let device_name = std::ffi::CStr::from_bytes_with_nul_unchecked(std::mem::transmute(
            &physical_device_properties.device_name[..],
        ))
        .to_str()?;

        log::info!("Initialised Vulkan with device {device_name} on queue family {compute_queue_family_index}");

        Ok(Context {
            entry,
            instance,
            device,
            queue,
            physical_device,
            command_pool,
            command_buffer,
        })
    }
}

fn main() -> anyhow::Result<()> {
    // Initialise our logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Create our Vulkan context (worth scrolling up to read the docs if you haven't)
    //
    // Note that *all* ash commands are unsafe; this is just a pragmatic response to the fact that
    // there's just no way to extend Rust's safety guarantees to the GPU.
    let context = unsafe { Context::new()? };

    // Grab a reference to our device, since we'll be using it a lot
    let device = &context.device;

    // Okay, so now that we've gone through all that singing and dancing, let's do some stuff.
    //
    // In this simple example we'll take some buffer of numbers, multiply them by two, store
    // the result in another buffer and print the result to the console. This sounds like a pretty
    // simple operation, right?
    let the_numbers: Vec<u32> = vec![10, 11, 13, 15, 16, 18, 34, 2]; // Powerball numbers from 03/10/2024

    // While the *compute* part of this is astonishingly simple as we'll see, the memory management
    // is *very* complicated. Vulkan, in general, tries to push these kinds of concerns back up
    // to the application developer so they have as much control as possible, but over time
    // it's become clear that a *little* bit of helper functionality is probably a good idea,
    // so Khronos members have contributed some libraries to help make that stuff simpler.
    //
    // I've shied away from these libraries myself because I've really wanted to understand the
    // fundamentals, but if you want to get something going quickly, it's probably a good idea to
    // just use an off-the shelf allocator.
    //
    // The allocator generally used is [AMD's VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator),
    // but this is written in CXX, and is therefore bad.
    //
    // Instead, we'll be using the aptly named
    // [gpu-allocator](https://github.com/Traverse-Research/gpu-allocator) provided to us by the
    // boffins at Traverse Research; a group of very clever graphics developers who've seen the
    // light and bring the magic of Rust to the world.

    // Let's start by creating the "allocator", which will, you know. Allocate.
    let mut allocator =
        gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
            instance: context.instance.clone(),
            device: context.device.clone(),
            physical_device: context.physical_device,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;

    // Great. Now let's create a buffer; this is basically just a handle to some memory with
    // some extra fanciness that the driver might use for optimisation. We ultimately want to
    // create two buffers here: an "input" buffer and an "output" buffer. Their uses are left
    // as an exercise to the reader.
    //
    // We must set usage flags to indicate to the driver how this buffer will be used. Here
    // we're just using storage buffers, the most generic kind, and flagging that we're going
    // to be using Buffer Device Address to get pointers to them.
    let buffer_size = the_numbers.len() * std::mem::size_of::<u32>();

    let input_buffer = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(buffer_size as vk::DeviceSize)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                ),
            None,
        )?
    };

    // Next we get the buffer's "requirements"; things like alignment, minimum allocation size
    // and so-on.
    let input_buffer_requirements = unsafe { device.get_buffer_memory_requirements(input_buffer) };

    // Now, let's allocate some memory for this buffer!
    //
    // Note that we want `CpuToGpu` for this buffer since we'll need to write the data to the GPU
    // from the CPU.
    let mut input_allocation =
        allocator.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "Input Buffer",
            requirements: input_buffer_requirements,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;

    // Now that we've allocated some memory, let's bind it to our buffer
    unsafe {
        device.bind_buffer_memory(
            input_buffer,
            input_allocation.memory(),
            input_allocation.offset(),
        )?
    };

    // Now here comes the fun part - we'll copy our numbers onto the GPU. We use the `presser`
    // crate for this because... well, `gpu-allocator` tells us to, whining about how blasting
    // bytes at the GPU is an "unsafe" operation, blah blah blah...
    presser::copy_from_slice_to_offset_with_align(
        &the_numbers[..],
        &mut input_allocation,
        0,
        input_buffer_requirements.alignment as _, // I *think* this is correct?
    )?;

    // Great! Now we'll great the output buffer to store our results.
    let output_buffer = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(buffer_size as vk::DeviceSize)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                ),
            None,
        )?
    };
    let output_buffer_requirements =
        unsafe { device.get_buffer_memory_requirements(output_buffer) };
    let mut output_allocation =
        allocator.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "Output Buffer",
            requirements: output_buffer_requirements,
            location: gpu_allocator::MemoryLocation::GpuToCpu, // Note we've flipped this
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;
    unsafe {
        device.bind_buffer_memory(
            output_buffer,
            output_allocation.memory(),
            output_allocation.offset(),
        )?
    };

    // OK. That's the hard part mostly done. Now we need to build up a pipeline to describe the
    // operations we'll be performing.
    //
    // You can think of a pipeline as the state the GPU needs to be in to execute some commands.
    // That includes things like shaders, layouts (below) and in the case of graphics, details on
    // how the hardware should be configured. Mercifully, we don't have to worry about *any* of
    // that extra crap with a compute shader.

    // First we have to describe the pipeline layout; you can think of this as your shader's ABI.
    // In our example, we'll just have a single push constant that stores the addresses of our
    // buffers
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&[
                vk::PushConstantRange::default()
                    .size(std::mem::size_of::<PushConstant>() as _)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ]),
            None,
        )?
    };

    // Next we need to load in our shader. We'll use Ralith's vk-shader-macros crate to make
    // this super easy.
    //
    // Note the return type of this function; it's a an array of u32s, not u8s, because:
    //
    // > SPIR-V is a stream of 32-bit words, not bytes, and this is reflected in APIs
    // > that consume it. In particular, passing a [u8] of SPIR-V that is not 4-byte-aligned
    // > to Vulkan is undefined behavior. Storing SPIR-V in its native format guarantees
    // > that this will never occur, without requiring copying or unsafe code.
    let shader_code = vk_shader_macros::include_glsl!("src/hello.glsl", kind: comp);

    // We jam this shader code into a "module", like so. Again, this is another example of
    // ash making life easier for us; the `code_size` member of vk::ShaderModuleCreateInfo
    // is automatically populated.
    let module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(shader_code),
            None,
        )?
    };

    // Great. Now, let's turn that into a pipeline.
    let pipeline = unsafe {
        device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .layout(pipeline_layout)
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .module(module)
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .name(c"main"),
                    )],
                None,
            )
            .unwrap()[0]
    };

    // Cool! Okay. Now, let's start submitting some commands! To do that, we'll use our trusty
    // command buffer that we allocated earlier.
    //
    // Note that since this is a simple handle, we don't need to take a reference to it, `Copy`
    // will just Do What We Want.
    let command_buffer = context.command_buffer;

    // First, we begin recording commands
    unsafe { device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())? };

    // Next, we bind to our pipeline
    unsafe {
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
    }

    // Then we create a push constant, populating the pointers to our buffers
    let push_constant = PushConstant {
        input_buffer: unsafe {
            device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(input_buffer),
            )
        },
        output_buffer: unsafe {
            device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(output_buffer),
            )
        },
        number_count: the_numbers.len() as _,
    };

    // Then we push those push constants onto the GPU
    unsafe {
        device.cmd_push_constants(
            command_buffer,
            pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            std::slice::from_raw_parts(
                &push_constant as *const _ as *const u8,
                std::mem::size_of::<PushConstant>(),
            ), // There's probably a better way of doing this, this is just how I do it.
        );
    }

    // Next, we "dispatch" some commands. This is a compute shader specific idea that just means
    // "run this shader" `n` times. `n` is calculated as:
    //
    // groupCountX × local_size_x × groupCountY × local_size_y × groupCountZ × local_size_z
    //
    // Where:
    // - groupCount is the size of the workgroups in each dimension
    // - local_size is the size of the "local" group in each dimension
    //
    // Workgroups allow fast, shared memory between compute shader invocations, and frankly
    // are a concept I've never really interacted with much. This is probably going to be an area
    // you'll need to research more! Here I'm just dispatching `the_numbers.len()` workgroups which
    // is probably super inefficient; who cares.
    //
    // Some jumping off points:
    //
    // - https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdDispatch.html
    // - https://registry.khronos.org/vulkan/specs/1.3/html/chap9.html#shaders-scope-workgroup
    //
    // And in particular, subgroups, which introduce cool features like balloting etc; I think this
    // will be very relevant to your interests.
    //
    // - https://www.khronos.org/blog/vulkan-subgroup-tutorial
    unsafe {
        device.cmd_dispatch(command_buffer, the_numbers.len() as _, 1, 1);
    }

    // And we're done! Declare the command buffer finished.
    unsafe {
        device.end_command_buffer(command_buffer)?;
    }

    // Now that we've done building up our command list, we submit the work to our queue. Before
    // we do, though, we want to create a "fence" that will allow us to wait for the operation to
    // complete. In real-time applications this is the stuff of nightmares, but for offline stuff,
    // who cares?
    //
    // NOTE:    I think there are invocation limits for shaders that you might need to work around,
    //          but those probably won't be an issue on high end NVIDIA hardware.
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;

    // Submit the work onto the queue
    unsafe {
        device.queue_submit(
            context.queue,
            &[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
            fence,
        )?;
    }

    let tick = std::time::Instant::now();
    log::info!("Work submitted, waiting..");

    // And now we play the waiting game!
    unsafe {
        // Here we're saying that we want to wait forever.
        device.wait_for_fences(&[fence], true, u64::MAX)?;
    }

    log::info!("Work complete. Waited for {}us", tick.elapsed().as_nanos());

    // Fabulous. Now, let's fetch the data back from the buffer!
    let slab = output_allocation
        .try_as_mapped_slab()
        .expect("Output data was not mapped?");

    // There's probably a safer way of doing this
    let data: &[u32] =
        unsafe { std::slice::from_raw_parts(slab.base_ptr() as *const u32, the_numbers.len()) };

    log::info!("Your numbers are: {data:?}. Verifying..");
    for (input, output) in the_numbers.into_iter().zip(data) {
        assert_eq!(*output, input * 2, "Invalid number");
    }
    log::info!(
        "Congratulations, you just burned some carbon for no good reason. I hope you're happy."
    );

    // Finally, free this memory so we don't get complaints about leaked memory.
    allocator.free(input_allocation)?;
    allocator.free(output_allocation)?;

    Ok(())
}

#[repr(C)]
#[derive(Debug, Clone)]
struct PushConstant {
    input_buffer: vk::DeviceAddress,
    output_buffer: vk::DeviceAddress,
    number_count: u32,
}
