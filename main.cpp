
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#pragma warning(disable:4996)
#endif

// OpenGL Graphics includes
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

#define REFRESH_DELAY	  10 //ms
#define REFRESH_DELAY_F	  20 //ms

void DeleteCudaLayers();
void InitCudaLayers();
void TimedFiledProc(int value);

unsigned int g_TotalErrors = 0;

// CheckFBO/BackBuffer class objects
CheckRender *g_CheckRender = NULL;

////////////////////////////////////////////////////////////////////////////////
// constants / global variables
unsigned int window_width = SWIDTH+CTRL_WIDTH;
unsigned int window_height = SHEIGHT;

unsigned int sim_width = SWIDTH;
unsigned int sim_height = SHEIGHT;

int iGLUTWindowHandle = 0;          // handle to the GLUT window
int iGLUTCtrlHandle = 0;
int iGLUTFieldHandle = 0;
int pickState  = 0;

unsigned int* cuda_dest_resource;
GLuint shDrawTex;  // draws a texture
struct cudaGraphicsResource *cuda_tex_result_resource;

GLuint fbo_source;
struct cudaGraphicsResource *cuda_tex_screen_resource;

unsigned int size_tex_data;
unsigned int num_texels;
unsigned int num_values;

// (offscreen) render target fbo variables
GLuint framebuffer;		// to bind the proper targets
GLuint depth_buffer;	// for proper depth test while rendering the scene
GLuint tex_screen;		// where we render the image
GLuint tex_cudaResult;	// where we will copy the CUDA result

int   *pArgc = NULL;
char **pArgv = NULL;

// Timer
static int fpsCount = 0;
static int fpsLimit = 1;
unsigned int timer;

extern GLuint shDrawPot;  // colors the teapot

////////////////////////////////////////////////////////////////////////////////
extern "C" void
launch_cudaProcess( dim3 grid, dim3 block, int sbytes, 
            cudaArray *g_data, unsigned int* g_odata, 
            int imgw, int imgh, int tilew, 
            int radius, float threshold, float highlight);

// Forward declarations
void runStdProgram(int argc, char** argv);
void FreeResource();
void Cleanup(int iExitCode);
void CleanupNoPrompt(int iExitCode);

// GL functionality
bool initCUDA( int argc, char **argv, bool bUseGL );
bool initGL(int *argc, char** argv);
void initGLBuffers();
void deleteFramebuffer(GLuint* fbo);
void initCUDABuffers();

// rendering callbacks
void displayField();
void keyboard(unsigned char key, int x, int y);
void reshape(int w, int h);
void mouseWheel(int, int, int, int);


const GLenum fbo_targets[] = {
  GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT, 
  GL_COLOR_ATTACHMENT2_EXT, GL_COLOR_ATTACHMENT3_EXT };  


void process( int width, int height, int radius);


// copy image and process using CUDA
void processImage();


// display image to the screen as textured quad
void displayImage(GLuint texture);


////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
bool pause_proc = false;

void displayField()
{
	if(pause_proc)
		return;

	glDisable(GL_MULTISAMPLE);

    cutStartTimer(timer);

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuffer);

	processImage();

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    displayImage(tex_cudaResult);

    // NOTE: I needed to add this call so the timing is consistent.
    // Need to investigate why
    cutilDeviceSynchronize();
    cutStopTimer(timer);

    // flip backbuffer
    glutSwapBuffers();
   
    // Update fps counter, fps/title display and log
    if (++fpsCount == fpsLimit) {
        char cTitle[256];
        float fps = 1000.0f / cutGetAverageTimerValue(timer);
        sprintf(cTitle, "CuMagic (%d x %d): %.1f fps", sim_width, sim_height, fps);  
        glutSetWindowTitle(cTitle);
        //shrLog("%s\n", cTitle);
		printf("%s\n", cTitle);
        fpsCount = 0; 
        fpsLimit = 10*(int)((fps > 1.0f) ? fps : 1.0f);
        cutResetTimer(timer);  
    }
}

void timerEvent(int value)
{
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}
void ActiveProc();
void DoBhv();

void TimedFiledProc(int value)
{
	if(!pause_proc)
	{
		ActiveProc();
		DoBhv();
	}

	glutSetWindow(iGLUTCtrlHandle);
    glutPostRedisplay();
	glutSetWindow(iGLUTFieldHandle);
    glutTimerFunc(REFRESH_DELAY_F, TimedFiledProc, 0);
}


////////////////////////////////////////////////////////////////////////////////
//! 
////////////////////////////////////////////////////////////////////////////////
void
deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);
    CUT_CHECK_ERROR_GL2();

    *tex = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! 
////////////////////////////////////////////////////////////////////////////////
void
deleteDepthBuffer(GLuint* depth)
{
    glDeleteRenderbuffersEXT(1, depth);
    CUT_CHECK_ERROR_GL2();

    *depth = 0;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
//	shrLogEx(LOGCONSOLE, 0, "", argv[0]); 
//    shrSetLogFileName ("CuPh.txt");
//    shrLog("%s Starting...\n\n", argv[0]);

	pArgc = &argc;
	pArgv = argv;

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if ( cutCheckCmdLineFlag(argc, (const char **)argv, "device")) {
		printf("[%s]\n", argv[0]);
		printf("   Does not explicitly support -device=n\n");
		printf("   This sample requires OpenGL.  Only -qatest and -glverify are supported\n");
		printf("exiting...\n");
        shrQAFinishExit(argc, (const char **)argv, QA_WAIVED);
	}



    runStdProgram(argc, argv);
    
//    shrEXIT(argc, (const char**)argv);
}

////////////////////////////////////////////////////////////////////////////////
//! 
////////////////////////////////////////////////////////////////////////////////
void FreeResource()
{
	cutilCheckError( cutDeleteTimer( timer ));

	// unregister this buffer object with CUDA
	cutilSafeCall(cudaGraphicsUnregisterResource(cuda_tex_screen_resource));

	cudaFree(cuda_dest_resource);

	deleteTexture(&tex_screen);
	deleteTexture(&tex_cudaResult);
	deleteDepthBuffer(&depth_buffer);
	deleteFramebuffer(&framebuffer);
	DeleteCudaLayers();
    cutilDeviceReset();
    if(iGLUTWindowHandle)glutDestroyWindow(iGLUTWindowHandle);

    // finalize logs and leave
    //shrLogEx(LOGBOTH | CLOSELOG, 0, "CuPh.exe Exiting...\n");
}

void Cleanup(int iExitCode)
{
    FreeResource();
    shrQAFinishExit(*pArgc, (const char **)pArgv, (iExitCode == EXIT_SUCCESS) ? QA_PASSED : QA_FAILED);
}

void CleanupNoPrompt(int iExitCode)
{
    FreeResource();
    printf("%s\n", (iExitCode == EXIT_SUCCESS) ? "PASSED" : "FAILED");
    exit(EXIT_SUCCESS);
}

void clickField(int button, int updown, int x, int y) ;
void motion (int x, int y);
void processMouseEntry(int state);

////////////////////////////////////////////////////////////////////////////////
//! Run standard demo loop with or without GL verification
////////////////////////////////////////////////////////////////////////////////
void
runStdProgram(int argc, char** argv)
{
    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    if( false == initGL(&argc, argv)) 
    {
        return;
    }

    // Now initialize CUDA context (GL context has been created already)
    initCUDA(argc, argv, true);

	InitCudaLayers();
    
    cutCreateTimer(&timer);
    cutResetTimer(timer);  

    // register callbacks
    glutDisplayFunc(displayField);
 //   glutKeyboardFunc(keyboard);
 //   glutReshapeFunc(reshape);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	glutTimerFunc(REFRESH_DELAY_F, TimedFiledProc, 0);

    glutMouseFunc(clickField);
    glutMotionFunc(motion);
	glutMouseWheelFunc(mouseWheel);

    initGLBuffers();

    initCUDABuffers();

    // start rendering mainloop
    glutMainLoop();

    Cleanup(EXIT_SUCCESS);

}

#include "layer_defines.h"

////////////////////////////////////////////////////////////////////////////////
//events handlers
//////////////////////
//////////////////////////////////////////////////////////
int debShowMax = 0;
void GetFieldData();

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) {
    case(27) :
        CleanupNoPrompt(EXIT_SUCCESS);
        break;
	case 'p':
		pause_proc = !pause_proc;
		break;
	case ' ':
		processImage();
		break;
	case 'd':
		GetFieldData();
		break;
	case 'm':
		debShowMax = (debShowMax+1)%2;
		break;
    }
}

void reshape(int w, int h)
{
    window_width = w;
    window_height = h;
	glutSetWindow(iGLUTFieldHandle);
}


static int clicked = 0;
int gMousex = 100, gMousey = 100;


int cursorArr[] = {
GLUT_CURSOR_RIGHT_ARROW,
GLUT_CURSOR_WAIT,//cicle
GLUT_CURSOR_TOP_SIDE,
GLUT_CURSOR_UP_DOWN,
GLUT_CURSOR_CYCLE, //cross-arrow
GLUT_CURSOR_CROSSHAIR,
};
int deltaGrab = 128;
int grabSet = 0;
extern float gObj0X, gObj0Y;
int gAction = DO_DEFAULT;

int block_click = 0;
int move_signal = 0;
void clickField(int button, int updown, int x, int y) {

	gMousex = x; gMousey = sim_height-y;

	if(pickState == 0 && clicked == 0 && updown == 0 && block_click == 0)
	{
		if(fabs(gObj0X-gMousex) < deltaGrab && fabs(gObj0Y-gMousey) < deltaGrab)
		{
			gMousex = gObj0X;
			gMousey = gObj0Y;
			glutWarpPointer(gMousex, sim_height -gMousey);
			move_signal = 1;
		}
		else
			gAction = DO_FORM0;
	}

	if(pickState == 1 && clicked == 0)
	{
		gAction = DO_VFILED0;
	}

	if(pickState == 2 && clicked == 0)
	{
		gAction = DO_VFILED1;
	}

	if(pickState == 3 && clicked == 0)
	{
		gAction = DO_VFILED2;
	}

	if(pickState == 4 && clicked == 0)
	{
		gAction = DO_VFILED3;
	}

	pickState = 0;
    clicked = !clicked;
	glutSetWindow(iGLUTFieldHandle);
	glutSetCursor(cursorArr[pickState]);

	block_click = max(block_click-1, 0);

	if(updown == 1)
		move_signal  = 0;


}



extern float moveLayerx;
extern float moveLayery;
void motion (int x, int y0) { 
    
	block_click = 0;

	float mousex = x;
	float mousey = sim_height-y0;

	if (clicked && move_signal == 1)
	{
		int ddx = mousex - gMousex;
		int ddy = mousey - gMousey;
		moveLayerx -= ddx;
		moveLayery -= ddy;
		gMousex = mousex; gMousey = mousey;

		if(pickState == 0)
		{
			gAction = DO_FORM0;
		}

		glutPostRedisplay();
    }



	if(fabs(gObj0X-mousex) < deltaGrab && fabs(gObj0Y-mousey) < deltaGrab)
	{
		glutSetCursor(cursorArr[5]);
		grabSet = 1;
	}
	else if(grabSet == 1)
	{
		glutSetCursor(cursorArr[0]);
		grabSet = 0;
	}
	
	glutSetWindow(iGLUTFieldHandle);
}

extern int hexClick;
extern int wmode;

void mouseWheel(int button, int dir, int x, int y)
{
//off
	return;
	if(hexClick)
	{
		wmode = (wmode+1)%2;
		glutSetWindow(iGLUTCtrlHandle);
	}
	else
	{
		if (dir > 0)
		{
			pickState = min(pickState+1, MAX_C_STATE);
		}
		else
		{
			pickState = max(pickState-1, MIN_C_STATE);
		}

		glutSetWindow(iGLUTFieldHandle);
		glutSetCursor(cursorArr[pickState]);
	}
    return;
}

void processMouseEntry(int state) {
	if (state == GLUT_LEFT)
	{
		//pickState = 0;
		glutSetCursor(cursorArr[pickState]);
	}
	else
	{
		
	}
	glutSetWindow(iGLUTFieldHandle);
}









