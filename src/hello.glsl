#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

// These layouts are essentially a mirror of what we declare on the CPU side. With a bit of effort,
// reflection can be performed upon the compiled SPIR-V to do this automagically.
layout(scalar, buffer_reference, buffer_reference_align = 8) readonly buffer Input
{
    uint numbers[];
};

layout(scalar, buffer_reference, buffer_reference_align = 8) writeonly buffer Output
{
    uint numbers[];
};

// Our push constant containing pointers to the above buffers
layout(scalar, push_constant) uniform Registers
{
    Input inputNumbers;
    Output outputNumbers;
    uint numberCount;
};

void main() {
    // Use the glsl builtin to find invocation ID in the x "dimension". Again, this is determined
    // through workgroup sizes and such; this will be important!
    uint index = gl_GlobalInvocationID.x;

    if (index >= numberCount)
    {
        return;
    }

    // Fetch our number from the input, double it, store it in the output.
    outputNumbers.numbers[index] = inputNumbers.numbers[index] * 2;
}
