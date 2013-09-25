#include <GL/glew.h>

// CUDA utilities and system includes
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <rendercheck_gl.h>
#include <cudaGL.h>

// Shared Library Test Functions
#include <shrUtils.h>
#include <shrQATest.h>
#include "layer_defines.h"

extern unsigned int* cuda_dest_resource;
extern GLuint shDrawTex;  // draws a texture
extern struct cudaGraphicsResource *cuda_tex_result_resource;

extern GLuint fbo_source;
extern struct cudaGraphicsResource *cuda_tex_screen_resource;

////////////////////////////////////////////////////////////////////////////////
//! Initialize CUDA context
////////////////////////////////////////////////////////////////////////////////
bool
initCUDA( int argc, char **argv, bool bUseGL )
{
    if (bUseGL) {
        if ( cutCheckCmdLineFlag(argc, (const char **)argv, "device"))
            cutilGLDeviceInit(argc, argv);
        else {
            cudaGLSetGLDevice (cutGetMaxGflopsDeviceId() );
        }
    } else {
        if ( cutCheckCmdLineFlag(argc, (const char **)argv, "device"))
            cutilDeviceInit(argc, argv);
        else {
            cudaSetDevice (cutGetMaxGflopsDeviceId() );
        }
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Allocate the "render target" of CUDA
////////////////////////////////////////////////////////////////////////////////
extern unsigned int size_tex_data;
extern unsigned int num_texels;
extern unsigned int num_values;

void initCUDABuffers()
{
    // set up vertex data parameter
    num_texels = sim_width * sim_height;
    num_values = num_texels * 4;
    size_tex_data = sizeof(GLubyte) * num_values;
    cutilSafeCall(cudaMalloc((void**)&cuda_dest_resource, size_tex_data));
}
