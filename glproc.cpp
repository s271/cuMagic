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
extern CheckRender *g_CheckRender;

////////////////////////////////////////////////////////////////////////////////
// constants / global variables
extern unsigned int window_width;
extern unsigned int window_height;
extern unsigned int sim_width;
extern unsigned int sim_height;
extern int iGLUTWindowHandle;          // handle to the GLUT window
extern int iGLUTCtrlHandle;          // handle to the GLUT window
extern int iGLUTFieldHandle;

extern unsigned int* cuda_dest_resource;
extern GLuint shDrawTex;  // draws a texture
extern struct cudaGraphicsResource *cuda_tex_result_resource;

extern GLuint fbo_source;
extern struct cudaGraphicsResource *cuda_tex_screen_resource;

extern unsigned int size_tex_data;
extern unsigned int num_texels;
extern unsigned int num_values;

extern int  g_Index ;
extern int  blur_radius;
extern int  max_blur_radius;

// (offscreen) render target fbo variables
extern GLuint framebuffer;		// to bind the proper targets
extern GLuint depth_buffer;	// for proper depth test while rendering the scene
extern GLuint tex_screen;		// where we render the image
extern GLuint tex_cudaResult;	// where we will copy the CUDA result


void createTextureDst(GLuint* tex_cudaResult, unsigned int size_x, unsigned int size_y);
void createTextureSrc(GLuint* tex_screen, unsigned int size_x, unsigned int size_y);
void deleteTexture(GLuint* tex);
void createDepthBuffer(GLuint* depth, unsigned int size_x, unsigned int size_y);
void deleteDepthBuffer(GLuint* depth);
void createFramebuffer(GLuint* fbo, GLuint color, GLuint depth);
void deleteFramebuffer(GLuint* fbo);
void process( int width, int height, int radius);

GLuint shDrawPot;  // colors the teapot
static const char *glsl_drawpot_fragshader_src = 
//WARNING: seems like the gl_FragColor doesn't want to output >1 colors...
//you need version 1.3 so you can define a uvec4 output...
//but MacOSX complains about not supporting 1.3 !!
// for now, the mode where we use RGBA8UI may not work properly for Apple : only RGBA16F works (default)

"#version 130\n"
"out uvec4 FragColor;\n"
"void main()\n"
"{"
"  FragColor = uvec4(gl_Color.xyz * 255.0, 255.0);\n"
"}\n";

static const char *glsl_drawtex_vertshader_src = 
        "void main(void)\n"
        "{\n"
        "	gl_Position = gl_Vertex;\n"
        "	gl_TexCoord[0].xy = gl_MultiTexCoord0.xy;\n"
        "}\n";

static const char *glsl_drawtex_fragshader_src = 
        "#version 130\n"
        "uniform usampler2D texImage;\n"
        "void main()\n"
        "{\n"
        "   vec4 c = texture(texImage, gl_TexCoord[0].xy);\n"
        "	gl_FragColor = c / 255.0;\n"
        "}\n";
void processLayer( int width, int height, unsigned int* cuda_int_dest);
void processImage()
{
  	processLayer(sim_width, sim_height, cuda_dest_resource);

    cudaArray *texture_ptr;
    cutilSafeCall(cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0));
    cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_result_resource, 0, 0));

    int num_texels = sim_width * sim_height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    cutilSafeCall(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dest_resource, size_tex_data, cudaMemcpyDeviceToDevice));

    cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0));

}

////////////////////////////////////////////////////////////////////////////////
//! 
////////////////////////////////////////////////////////////////////////////////
void
createTextureSrc(GLuint* tex_screen, unsigned int size_x, unsigned int size_y)
{
    // create a texture
    glGenTextures(1, tex_screen);
    glBindTexture(GL_TEXTURE_2D, *tex_screen);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // buffer data
    //shrLog("Creating a Texture render target GL_RGBA16F_ARB\n");
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, size_x, size_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    CUT_CHECK_ERROR_GL2();
    // register this texture with CUDA
    cutilSafeCall(cudaGraphicsGLRegisterImage(&cuda_tex_screen_resource, *tex_screen, 
                          GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
}
////////////////////////////////////////////////////////////////////////////////
//! 
////////////////////////////////////////////////////////////////////////////////
void
createTextureDst(GLuint* tex_cudaResult, unsigned int size_x, unsigned int size_y)
{
    // create a texture
    glGenTextures(1, tex_cudaResult);
    glBindTexture(GL_TEXTURE_2D, *tex_cudaResult);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, size_x, size_y, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
    CUT_CHECK_ERROR_GL2();
    // register this texture with CUDA
    cutilSafeCall(cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, *tex_cudaResult, 
                          GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
}
////////////////////////////////////////////////////////////////////////////////
//! 
////////////////////////////////////////////////////////////////////////////////
void
createDepthBuffer(GLuint* depth, unsigned int size_x, unsigned int size_y)
{
    // create a renderbuffer
    glGenRenderbuffersEXT(1, depth);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, *depth);

    // allocate storage
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, size_x, size_y);

    // clean up
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);

    CUT_CHECK_ERROR_GL2();
}
////////////////////////////////////////////////////////////////////////////////
//! 
////////////////////////////////////////////////////////////////////////////////
//void
//createRenderBuffer(GLuint* render, unsigned int size_x, unsigned int size_y)
//{
//    // create a renderbuffer
//    glGenRenderbuffersEXT(1, render);
//    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, *render);
//
//    // allocate storage
//    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_RGBA8, size_x, size_y);
//
//    // clean up
//    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
//
//    CUT_CHECK_ERROR_GL2();
//
//	cutilSafeCall(cudaGraphicsGLRegisterImage(&cuda_tex_screen_resource, *render, 
//					      GL_RENDERBUFFER_EXT, cudaGraphicsMapFlagsReadOnly));
//}
////////////////////////////////////////////////////////////////////////////////
//! 
////////////////////////////////////////////////////////////////////////////////
void
createFramebuffer(GLuint* fbo, GLuint color, GLuint depth)
{
    // create and bind a framebuffer
    glGenFramebuffersEXT(1, fbo);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, *fbo);

    // attach images
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, color, 0);
    //glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_RENDERBUFFER_EXT, color);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depth);

    // clean up
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    CUT_CHECK_ERROR_GL2();
}

////////////////////////////////////////////////////////////////////////////////
//! 
////////////////////////////////////////////////////////////////////////////////
void
deleteFramebuffer( GLuint* fbo)
{
    glDeleteFramebuffersEXT(1, fbo);
    CUT_CHECK_ERROR_GL2();

    *fbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
// rendering callbacks
void keyboard(unsigned char key, int x, int y);
void mouseWheel(int, int, int, int);
void clickField(int button, int updown, int x, int y) ;
void clickCtrl(int button, int updown, int x, int y);
void reshape(int w, int h);
void motion (int x, int y);
void processMouseEntry(int state);
void displayCtrls();
void motionCtrl(int x, int y);

#include "glcontrols.h"


bool SetupViewGL(int w, int h)
{
    glewInit();
    // initialize necessary OpenGL extensions
    //if (! glewIsSupported(
    //    "GL_VERSION_2_0 " 
    //    "GL_ARB_pixel_buffer_object "
    //    "GL_EXT_framebuffer_object "
    //    )) {
    //    shrLog("ERROR: Support for necessary OpenGL extensions missing.");
    //    fflush(stderr);
    //    return CUTFalse;
    //}

    glClearColor(0.5, 0.5, 0.5, 1.0);

    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, w, h);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)w / (GLfloat) h, 0.1f, 10.0f);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glEnable(GL_LIGHT0);
    float red[] = { 1.0f, 0.1f, 0.1f, 1.0f };
    float white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0f);

    CUT_CHECK_ERROR_GL2();

	return CUTTrue;
}

const char *sSDKname = "txtGL";
bool initGL(int *argc, char **argv )
{
	controls.initCtrls();
	// Create GL context
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);

    glutInitWindowSize(window_width, window_height);
    iGLUTWindowHandle = glutCreateWindow("CuMagicField");
//-----------------------------------------------------------
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
	glutMouseWheelFunc(mouseWheel);
	glutEntryFunc(processMouseEntry);
//-----------------------------------------------------------
	iGLUTCtrlHandle = glutCreateSubWindow(iGLUTWindowHandle, sim_width+9, 9, CTRL_WIDTH, sim_height);
	bool res = SetupViewGL(CTRL_WIDTH, sim_height);
	
	glutDisplayFunc(displayCtrls);
	glutMouseFunc(clickCtrl);
	//glutMotionFunc(motionCtrl);
	//glutMouseWheelFunc(mouseWheelCtrl);
	glutPassiveMotionFunc(motionCtrl);
	glutPostRedisplay();

	if(!res)
		return false;

	iGLUTFieldHandle = glutCreateSubWindow(iGLUTWindowHandle, 9, 9, sim_width, sim_height);
	res = SetupViewGL(sim_width, sim_height);

    return res;
}

////////////////////////////////////////////////////////////////////////////////
//! 
////////////////////////////////////////////////////////////////////////////////
GLuint compileGLSLprogram(const char *vertex_shader_src, const char *fragment_shader_src)
{
  GLuint v, f, p = 0;

  p = glCreateProgram();
    
  if (vertex_shader_src) {
      v = glCreateShader(GL_VERTEX_SHADER);
      glShaderSource(v, 1, &vertex_shader_src, NULL);
      glCompileShader(v);

      // check if shader compiled
      GLint compiled = 0;
      glGetShaderiv(v, GL_COMPILE_STATUS, &compiled);
      
      if (!compiled)
      {
          //#ifdef NV_REPORT_COMPILE_ERRORS
          char temp[256] = "";
          glGetShaderInfoLog( v, 256, NULL, temp);
          //shrLog("Vtx Compile failed:\n%s\n", temp);
          //#endif
          glDeleteShader( v);
          return 0;
      }
      else
      glAttachShader(p,v);
  }
  
  if (fragment_shader_src)  {
      f = glCreateShader(GL_FRAGMENT_SHADER);
      glShaderSource(f, 1, &fragment_shader_src, NULL);
      glCompileShader(f);

      // check if shader compiled
      GLint compiled = 0;
      glGetShaderiv(f, GL_COMPILE_STATUS, &compiled);
      
      if (!compiled)
      {
          //#ifdef NV_REPORT_COMPILE_ERRORS
          char temp[256] = "";
          glGetShaderInfoLog(f, 256, NULL, temp);
          //shrLog("frag Compile failed:\n%s\n", temp);
          //#endif
          glDeleteShader(f);
          return 0;
      }
      else
      glAttachShader(p,f);
  }
  
  glLinkProgram(p);

  int infologLength = 0;
  int charsWritten  = 0;
  
  glGetProgramiv(p, GL_INFO_LOG_LENGTH, (GLint *)&infologLength);
  
  if (infologLength > 0) {
      char *infoLog = (char *)malloc(infologLength);
      glGetProgramInfoLog(p, infologLength, (GLsizei *)&charsWritten, infoLog);
      //shrLog("Shader compilation error: %s\n", infoLog);
      free(infoLog);
  }

  return p;
}


////////////////////////////////////////////////////////////////////////////////
//! 
////////////////////////////////////////////////////////////////////////////////
void initGLBuffers()
{
   
    // create texture that will receive the result of CUDA
    createTextureDst(&tex_cudaResult, sim_width, sim_height);

    // create texture for blitting onto the screen
    createTextureSrc(&tex_screen, sim_width, sim_height);
    //createRenderBuffer(&tex_screen, sim_width, sim_height); // Doesn't work
    
    // create a depth buffer for offscreen rendering
    createDepthBuffer(&depth_buffer, sim_width, sim_height);
    
    // create a framebuffer for offscreen rendering
    createFramebuffer(&framebuffer, tex_screen, depth_buffer);
    
    // load shader programs
    shDrawPot = compileGLSLprogram(NULL, glsl_drawpot_fragshader_src);


    shDrawTex = compileGLSLprogram(glsl_drawtex_vertshader_src, glsl_drawtex_fragshader_src);


    CUT_CHECK_ERROR_GL2();
}

// display image to the screen as textured quad
void displayImage(GLuint texture)
{
    glBindTexture(GL_TEXTURE_2D, texture);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode( GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, sim_width, sim_height);

    // if the texture is a 8 bits UI, scale the fetch with a GLSL shader

    glUseProgram(shDrawTex);
    GLint id = glGetUniformLocation(shDrawTex, "texImage");
    glUniform1i(id, 0); // texture unit 0 to "texImage"
    CUT_CHECK_ERROR_GL2();

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 0.0); glVertex3f(1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 1.0, 0.5);
    glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, 0.5);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);


    glUseProgram(0);

    CUT_CHECK_ERROR_GL2();
}

