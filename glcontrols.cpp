#include <GL/glew.h>
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <rendercheck_gl.h>

#include "glcontrols.h"

extern int iGLUTCtrlHandle; 
extern int iGLUTFieldHandle;

CtrlsSinglet controls;
GeomStorage geomStorage;
#pragma warning(disable: 4244)
//#define RED 0
//#define YELLOW 1
//#define VIOLET 2
//#define BLUE 3
//#define GREEN 4
//#define ORANGE 5
// magneta 6
// jade 7

float ColorsHex[] =
{
1, 0, 0,
1, 1, 0,
.6f, .2f, .8f,
0, 0, 1,
0, .8f, 0.0,
.8f, .4f, .1f,
.9f, .1f, 1.f,
.0f, .6f, .5f,
.2f, .2f, .2f,
0, 0, 0
};

int C2P[] = 
{
1, //red
4, //yellow
6, //violet
2,//blue
3, //green
5, //orange
8, //magneta
7, //jade
0, //black
};

int P2C[] = 
{
8,//blank
0, //red
3,//blue
4, //green
1, //yellow
5, //orange
2, //violet
7, //magneta
6, //jade
};

extern float gIntegralScale;

void DrawCircle(float x0, float y0, int rad)
{
	static vector2 circle[CIRCL_SEG+1]; 
	circle[0].x = x0;
	circle[0].y = y0;

	for(int i = 0; i < CIRCL_SEG; i++)
	{
		circle[i+1] = circle[0] + rad*geomStorage.circle[i];
	}

	glBegin(GL_TRIANGLES);
	for(int i = 0; i < CIRCL_SEG; i++)
	{
		glVertex3f(circle[0].x, circle[0].y, 0.5);
		glVertex3f(circle[i+1].x, circle[i+1].y, 0.5);
		glVertex3f(circle[(i+1)%CIRCL_SEG+1].x, circle[(i+1)%CIRCL_SEG+1].y, 0.5);
	}
	glEnd();
}


void DrawRing(float x0, float y0, int rad, int radi)
{
	static vector2 circle[CIRCL_SEG*2]; 
	vector2 cnt;
	cnt.x = x0;
	cnt.y = y0;

	for(int i = 0; i < CIRCL_SEG; i++)
	{
		circle[i] = cnt + radi*geomStorage.circle[i];
		circle[i+CIRCL_SEG] = cnt + rad*geomStorage.circle[i];
	}

	glBegin(GL_QUADS);
	for(int i = 0; i < CIRCL_SEG; i++)
	{
		glVertex3f(circle[i].x, circle[i].y, 0.5);
		glVertex3f(circle[(i+1)%CIRCL_SEG].x, circle[(i+1)%CIRCL_SEG].y, 0.5);
		glVertex3f(circle[(i+1)%CIRCL_SEG + CIRCL_SEG].x, circle[(i+1)%CIRCL_SEG + CIRCL_SEG].y, 0.5);
		glVertex3f(circle[i + CIRCL_SEG].x, circle[i + CIRCL_SEG].y, 0.5);
	}
	glEnd();
}

void DrawHex(vector2* hex)
{
    glBegin(GL_LINE_STRIP);
        glVertex3f(hex[0].x, hex[0].y, 0.5);      
        glVertex3f(hex[2].x, hex[2].y, 0.5);      
        glVertex3f(hex[4].x, hex[4].y, 0.5);  
		glVertex3f(hex[0].x, hex[0].y, 0.5);
    glEnd();
    glBegin(GL_LINE_STRIP);
        glVertex3f(hex[1].x, hex[1].y, 0.5);      
        glVertex3f(hex[3].x, hex[3].y, 0.5);      
        glVertex3f(hex[5].x, hex[5].y, 0.5);  
		glVertex3f(hex[1].x, hex[1].y, 0.5);
    glEnd();
}

void DrawMinihex(int k)
{
    glBegin(GL_LINE_STRIP);
        glVertex3f(controls.minihex[k][0].x, controls.minihex[k][0].y, 0.5);      
        glVertex3f(controls.minihex[k][2].x, controls.minihex[k][2].y, 0.5);      
        glVertex3f(controls.minihex[k][4].x, controls.minihex[k][4].y, 0.5);  
		glVertex3f(controls.minihex[k][0].x, controls.minihex[k][0].y, 0.5);
    glEnd();
    glBegin(GL_LINE_STRIP);
        glVertex3f(controls.minihex[k][1].x, controls.minihex[k][1].y, 0.5);      
        glVertex3f(controls.minihex[k][3].x, controls.minihex[k][3].y, 0.5);      
        glVertex3f(controls.minihex[k][5].x, controls.minihex[k][5].y, 0.5);  
		glVertex3f(controls.minihex[k][1].x, controls.minihex[k][1].y, 0.5);
    glEnd();
}

void DrawLine(float x0, float y0, float x1, float y1)
{
	vector2 v0(x0, y0);
	vector2 v1(x1, y1);

	glBegin(GL_LINES);
		glVertex3f(v0.x, v0.y, 0.5);      
		glVertex3f(v1.x, v1.y, 0.5); 
	glEnd();
}

void DrawZZLine(float x0, float y0, float x1, float y1)
{
	vector2 v0(x0, y0);
	vector2 v1(x1, y1);

	vector2 v = v1 - v0;
	vector2 vortho(v.y, - v.x);

	float lenv = sqrtf(v.x*v.x + v.y*v.y);
	float norm =  1.f/lenv;

	vortho = norm*vortho;
	v = norm*v;

	vector2 va1 = v1 + 3*vortho - 18*v;
	vector2 va2 = v1 + -3*vortho - 18*v;
	float zigsize = 16;
	float leneff = lenv-20;
	int nzigs = leneff/zigsize;
	vector2 vs = v0 + 10*v;
	float hzig = zigsize/4;

	glBegin(GL_LINE_STRIP);
		glVertex3f(v0.x, v0.y, 0.5);   
		glVertex3f(vs.x, vs.y, 0.5);
		for(int i = 0; i < nzigs; i++)
		{
			vector2 vsc = vs + (i*zigsize)*v;
			glVertex3f(vsc.x, vsc.y, 0.5);
			vector2 vc = vsc + hzig*v + hzig*vortho;
			glVertex3f(vc.x, vc.y, 0.5);
			vc = vsc + 3*hzig*v - hzig*vortho;
			glVertex3f(vc.x, vc.y, 0.5);
			vc = vsc + 4*hzig*v;
			glVertex3f(vc.x, vc.y, 0.5);
		}
		glVertex3f(v1.x, v1.y, 0.5);
	glEnd();
}


void DrawArrow(float x0, float y0, float x1, float y1)
{
	vector2 v0(x0, y0);
	vector2 v1(x1, y1);

	vector2 v = v1 - v0;
	vector2 vortho(v.y, - v.x);
	float norm =  1./sqrtf(v.x*v.x + v.y*v.y);

	vector2 va1 = v1 + norm*(3*vortho - 18*v);
	vector2 va2 = v1 + norm*(-3*vortho - 18*v);

	glBegin(GL_LINES);
		glVertex3f(v0.x, v0.y, 0.5);      
		glVertex3f(v1.x, v1.y, 0.5); 

		glVertex3f(v1.x, v1.y, 0.5);      
		glVertex3f(va1.x, va1.y, 0.5); 

		glVertex3f(v1.x, v1.y, 0.5);      
		glVertex3f(va2.x, va2.y, 0.5); 
	glEnd();
}

void DrawZZArrow(float x0, float y0, float x1, float y1)
{
	vector2 v0(x0, y0);
	vector2 v1(x1, y1);

	vector2 v = v1 - v0;
	vector2 vortho(v.y, - v.x);

	float lenv = sqrtf(v.x*v.x + v.y*v.y);
	float norm =  1./lenv;

	vortho = norm*vortho;
	v = norm*v;

	vector2 va1 = v1 + 3*vortho - 18*v;
	vector2 va2 = v1 + -3*vortho - 18*v;
	float zigsize = 16;
	float leneff = lenv-18-20;
	int nzigs = leneff/zigsize;
	vector2 vs = v0 + 10*v;
	float hzig = zigsize/4;

	glBegin(GL_LINE_STRIP);
		glVertex3f(v0.x, v0.y, 0.5);   
		glVertex3f(vs.x, vs.y, 0.5);
		for(int i = 0; i < nzigs; i++)
		{
			vector2 vsc = vs + (i*zigsize)*v;
			glVertex3f(vsc.x, vsc.y, 0.5);
			vector2 vc = vsc + hzig*v + hzig*vortho;
			glVertex3f(vc.x, vc.y, 0.5);
			vc = vsc + 3*hzig*v - hzig*vortho;
			glVertex3f(vc.x, vc.y, 0.5);
			vc = vsc + 4*hzig*v;
			glVertex3f(vc.x, vc.y, 0.5);
		}
		glVertex3f(v1.x, v1.y, 0.5);
	glEnd();


	glBegin(GL_LINES);
		glVertex3f(v1.x, v1.y, 0.5);      
		glVertex3f(va1.x, va1.y, 0.5); 

		glVertex3f(v1.x, v1.y, 0.5);      
		glVertex3f(va2.x, va2.y, 0.5); 
	glEnd();
}

void DrawRect(float x0, float y0, float x1, float y1)
{
	glBegin(GL_QUADS);
	glVertex3f(x0, y0, 0.5);
	glVertex3f(x1, y0, 0.5);
	glVertex3f(x1, y1, 0.5);
	glVertex3f(x0, y1, 0.5);
	glEnd();
}


bool ctrlClicked = false;

int hexClick = 0;
int clickSrc =-1;
int clickClr =-1;
int glLineDstx = -1;
int glLineDsty = -1;
int wmode = 0;
int currentConnection = 1;

extern GSum gFiledIntegral;
void DrawRBbar()
{
	float red = gFiledIntegral.v[P_RED]/gIntegralScale;
	float blue = gFiledIntegral.v[P_BLUE]/gIntegralScale;
	float maxsum = max(red+blue, 1);
	red /= maxsum;
	blue /= maxsum;

	float xs = 10;
	float ys = CTRL_WIDTH-20;
	float len = (CTRL_WIDTH-20);

	glColor3fv(ColorsHex + 3*RED);
	DrawRect(xs, ys, xs + len*red, ys + 5);

	glColor3fv(ColorsHex + 3*BLUE);
	DrawRect(xs + (1-blue)*len, ys, xs + len, ys + 5);

}

void displayUpperCtrls()
{
	for( int k = 0; k < MAX_INTERACTIONS; k++)
	{
		glColor3fv (ColorsHex + 3*k);
		DrawMinihex(k);
	}

	if(hexClick)
	{
		if(hexClick == 1)
			glColor3f(.6f, .6f, .6f);
		else
			glColor3fv(ColorsHex + 3*clickClr);

		if(wmode)
			DrawZZLine(controls.bigHex[clickSrc].x, controls.bigHex[clickSrc].y, glLineDstx, glLineDsty);
		else
			DrawLine(controls.bigHex[clickSrc].x, controls.bigHex[clickSrc].y, glLineDstx, glLineDsty);
	}

	for(int k = 0; k < MAX_INTERACTIONS; k++)
	{
		for(int i = 0; i < MAX_INTERACTIONS; i++)
		{
			if(controls.interactMatrixF[k][i])
			{
				int ival = controls.interactMatrixF[k][i]-1;
				glColor3fv(ColorsHex + 3*ival);
				DrawLine(controls.bigHex[k].x, controls.bigHex[k].y, controls.bigHex[i].x, controls.bigHex[i].y); 
			}
			if(controls.interactMatrixP[k][i])
			{
				int ival = controls.interactMatrixP[k][i]-1;
				glColor3fv(ColorsHex + 3*ival);
				DrawZZLine(controls.bigHex[k].x, controls.bigHex[k].y, controls.bigHex[i].x, controls.bigHex[i].y); 
			}


		}
	}

	DrawRBbar();
}

void displayVertical();
extern bool pause_proc;
void displayCtrls()
{
	if(pause_proc)
		return;

	glClearColor(0.8f, 0.8f, 0.8f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_MULTISAMPLE);

    glEnable( GL_LINE_SMOOTH );
    glHint( GL_LINE_SMOOTH_HINT, GL_NICEST );

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, CTRL_WIDTH, sim_height, 0, -1.0, 1.0);
    glMatrixMode( GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, CTRL_WIDTH, sim_height);

	glLineWidth(1.5);

	displayUpperCtrls();

	displayVertical();

	glutSwapBuffers();
}


void initGeom()
{

	for(int i = 0; i < 6; i++)
	{
		float a = -M_PI/2 + i*M_PI/3;
		geomStorage.star6[i].x = cosf(a);
		geomStorage.star6[i].y = sinf(a);
	}
//circle
	for(int i = 0; i < CIRCL_SEG; i++)
	{
		float a = -M_PI/2 + i*2*M_PI/CIRCL_SEG;
		geomStorage.circle[i].x = cosf(a);
		geomStorage.circle[i].y = sinf(a);
	}
}

void CtrlsSinglet::initCtrls()
{
	initGeom();

	memset(interactMatrixF, 0, sizeof(interactMatrixF));
	memset(interactMatrixP, 0, sizeof(interactMatrixP));

	bigHexCnt.x = CTRL_WIDTH/2;
	bigHexCnt.y = CTRL_WIDTH/2-10;

	float rad = 90;

	for(int k = 0; k < 6; k++)
		bigHex[k] = controls.bigHexCnt + rad*geomStorage.star6[k];

	bigHex[6] =  bigHexCnt + vector2(-rad*.25f, -rad*.3f);
	bigHex[7] =   bigHexCnt + vector2(rad*.25f, -rad*.3f);

	bigHex[MAX_INTERACTIONS-1] = bigHexCnt;

	for(int k = 0; k < MAX_INTERACTIONS; k++)
	for(int i = 0; i < 6; i++)
		minihex[k][i] = controls.bigHex[k] + 12*geomStorage.star6[i];

}

#include "layer_defines.h"
#include "bhv.h"
void InitCardPars(int &vstep , int &hcenter, int &acenter);
void ArrayCleanup(BoneSet* boneSet, int& setSize);

void clickCtrl(int button, int updown, int x, int y)
{

	if(!ctrlClicked)
	{
		for(int k = 0; k < MAX_INTERACTIONS; k++)
		{
			if(abs(x -controls.bigHex[k].x) < 15 && abs(y -controls.bigHex[k].y) < 15)
			{
				if(hexClick == 0)
				{
					clickSrc = k;
					glLineDstx = controls.bigHex[k].x;
					glLineDsty = controls.bigHex[k].y;
					hexClick = 1;
					break;
				}

				if(hexClick == 1)
				{
					clickClr = k;	
					hexClick = 2;
					break;
				}

				if(hexClick == 2)
				{
					int newVal = clickClr + 1;
					if(clickClr == BLACK)
						newVal = 0;

					if(wmode)
					{
						controls.interactMatrixP[clickSrc][k] = newVal;
						//symmetry
						controls.interactMatrixP[k][clickSrc] = newVal;
					}
					else
						controls.interactMatrixF[clickSrc][k] = newVal;
					clickClr = -1;
					hexClick = 0;
					break;
				}
			}
		}
		if(gActiveSets.statusPickRed >= 1)
		{
			int vstep;
			int hcenter;
			int acenter;
			InitCardPars(vstep , hcenter, acenter);
			int pos_count = 0;
			for(int i =0; i < gActiveSets.redSetSize; i++)
			{

				if(gActiveSets.redBoneSet[i].bhv_ptr == NULL || gActiveSets.redBoneSet[i].num <= 0)
					continue;

				int yc = CTRL_WIDTH + vstep/2 + pos_count*vstep;
				pos_count++;

				if(abs(x - hcenter) < vstep/2 && abs(y - yc) < vstep/2)
				{
					gActiveSets.redBoneSet[i].bhv_ptr->Process();
					gActiveSets.redBoneSet[i].num--;
					ArrayCleanup(gActiveSets.redBoneSet, gActiveSets.redSetSize);
					gActiveSets.statusPickRed -= 1;
					break;
				}

			}
		}

	}

	ctrlClicked = !ctrlClicked;
	glutPostRedisplay();
	glutSetWindow(iGLUTFieldHandle);
}

void motionCtrl(int x, int y)
{
	if(hexClick)
	{
		glLineDstx = x;
		glLineDsty = y;
		glutPostRedisplay();
	}
	glutSetWindow(iGLUTFieldHandle);
};