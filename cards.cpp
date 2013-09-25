// CUDA utilities and system includes
#include <GL/glew.h>
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <rendercheck_gl.h>

#include "glcontrols.h"
// Shared Library Test Functions
#include <shrUtils.h>
#include "layer_defines.h"
#include "bhv.h"




extern GeomStorage geomStorage;
extern float ColorsHex[];

ActiveSets gActiveSets;
float deltaPick = .014f;


void DrawCircle(float x0, float y0, int rad);
void DrawRing(float x0, float y0, int rad, int radi);
void DrawRect(float x0, float y0, float x1, float y1);
void DrawHex(vector2* hex);
void DrawLine(float x0, float y0, float x1, float y1);

#pragma warning(disable: 4244)

void RenderLink(int src, int conn, int dst, int xc, int yc)
{
	vector2 hex[6];
	int gap = 12;

	vector2 lefthex(xc - (CTRL_WIDTH/8 -gap), yc);

	for(int i=0; i < 6; i++)
		hex[i] = lefthex + 12*geomStorage.star6[i];

	glColor3fv (ColorsHex + 3*src);
	DrawHex(hex);


	vector2 righthex(xc + (CTRL_WIDTH/8 -gap), yc);

	for(int i=0; i < 6; i++)
		hex[i] = righthex + 12*geomStorage.star6[i];

	glColor3fv (ColorsHex + 3*dst);
	DrawHex(hex);

	glColor3fv (ColorsHex + 3*conn);
	DrawLine(lefthex.x, lefthex.y, righthex.x, righthex.y);
}

void LinkElement::Render(int x, int y)
{
	RenderLink( P2C[boneId.src], P2C[boneId.connect], P2C[boneId.dst], x,y);
}

void GreenBlobHeuristic::Render(int x, int y)
{
	int vheight = SHEIGHT - CTRL_WIDTH;
	int vstep = vheight/MAX_SET_SIZE;
	int rad = vstep/2*.7f;
	glColor3fv(ColorsHex + 3*GREEN);
	DrawCircle(x, y, rad);
}


void YellowBlobElement::Render(int x, int y)
{
	int vheight = SHEIGHT - CTRL_WIDTH;
	int vstep = vheight/MAX_SET_SIZE;
	int rad = vstep/2*.7f;
	glColor3fv(ColorsHex + 3*YELLOW);
	DrawCircle(x, y, rad);
}

void JadeRingHeuristic::Render(int x, int y)
{
	int vheight = SHEIGHT - CTRL_WIDTH;
	int vstep = vheight/MAX_SET_SIZE;
	int rad = vstep/2*.7f;
	int radi = vstep/2*.5f;
	glColor3fv(ColorsHex + 3*JADE);
	DrawRing(x, y, rad, radi);
}

void MagnetaRingElement::Render(int x, int y)
{
	int vheight = SHEIGHT - CTRL_WIDTH;
	int vstep = vheight/MAX_SET_SIZE;
	int rad = vstep/2*.7f;
	int radi = vstep/2*.5f;
	glColor3fv(ColorsHex + 3*MAGNETA);
	DrawRing(x, y, rad, radi);
}

int gap = 16;

void DrawCardStatus()
{
	float vstart = CTRL_WIDTH+gap;
	float vscale = (SHEIGHT - CTRL_WIDTH - 2*gap);
	glColor3fv(ColorsHex + 3*RED);
	float valR = vscale*gActiveSets.statusPickRed;
	DrawRect(gap, vstart, gap+3, vstart + valR);

	glColor3fv(ColorsHex + 3*BLUE);
	float valB = vscale*gActiveSets.statusPickBlue;
	DrawRect(CTRL_WIDTH-gap-3, vstart, CTRL_WIDTH-gap, vstart + valB);

}; 

void InitCardPars(int &vstep , int &hcenter, int &acenter)
{
	int vheight = SHEIGHT - CTRL_WIDTH;
	vstep = vheight/MAX_SET_SIZE;
	int xcellsize = (CTRL_WIDTH-2*gap)/2;
	hcenter = gap + xcellsize/2;
	acenter = gap + 3*xcellsize/2;
}

void displayVertical()
{
	int vstep;
	int hcenter;
	int acenter;
	InitCardPars(vstep , hcenter, acenter);
	
	int pos_count = 0;
	for(int i =0; i < gActiveSets.blueSetSize; i++)
	{

		if(gActiveSets.blueBoneSet[i].bhv_ptr == NULL || gActiveSets.blueBoneSet[i].num <= 0)
			continue;

		int yc = CTRL_WIDTH + vstep/2 + pos_count*vstep;

		gActiveSets.blueBoneSet[i].bhv_ptr->Render(acenter, yc);

		pos_count++;
	}


	pos_count = 0;
	for(int i =0; i < gActiveSets.redSetSize; i++)
	{
		if(gActiveSets.redBoneSet[i].bhv_ptr == NULL || gActiveSets.redBoneSet[i].num <= 0)
			continue;

		int yc = CTRL_WIDTH + vstep/2 + pos_count*vstep;

		gActiveSets.redBoneSet[i].bhv_ptr->Render(hcenter, yc);

		pos_count++;
	}

	DrawCardStatus();

};
//--------------------------------------
LinkElement redLinkElement[MAX_DECK];
extern BhvEelemntAbstract* gElementPtr[MAX_BONE_ELEMNTS];
extern int gNumElements;

YellowBlobElement yellowBlobElement;
MagnetaRingElement magnetaRingElement;

int startRed = 0;
int sizeRed = 0;
void InitRedBoneBhv()
{
	startRed = gNumElements;
	int redLinks =0;
	redLinkElement[redLinks].SetBone(P_RED, P_ORANGE, P_BLACK);//0
	gElementPtr[gNumElements] = redLinkElement + redLinks;
	redLinks++;
	gNumElements++;

	redLinkElement[redLinks].SetBone(P_ORANGE, P_RED, P_BLACK);//1
	gElementPtr[gNumElements] = redLinkElement + redLinks;
	redLinks++;
	gNumElements++;

	redLinkElement[redLinks].SetBone(P_YELLOW, P_RED, P_BLACK);//2
	gElementPtr[gNumElements] = redLinkElement + redLinks;
	redLinks++;
	gNumElements++;

	redLinkElement[redLinks].SetBone(P_YELLOW, P_ORANGE, P_BLACK);//3
	gElementPtr[gNumElements] = redLinkElement + redLinks;
	redLinks++;
	gNumElements++;

	redLinkElement[redLinks].SetBone(P_RED, P_MAGNETA, P_BLACK);//4
	gElementPtr[gNumElements] = redLinkElement + redLinks;
	redLinks++;
	gNumElements++;

	redLinkElement[redLinks].SetBone(P_YELLOW, P_BLACK, P_BLACK);//5
	gElementPtr[gNumElements] = redLinkElement + redLinks;
	redLinks++;
	gNumElements++;

	redLinkElement[redLinks].SetBone(P_RED, P_BLACK, P_BLACK);//5
	gElementPtr[gNumElements] = redLinkElement + redLinks;
	redLinks++;
	gNumElements++;

	gElementPtr[gNumElements] = &yellowBlobElement;//5
	gNumElements++;

	gElementPtr[gNumElements] = &magnetaRingElement;//6
	gNumElements++;

	sizeRed = gNumElements-startRed;
}
//-------------------------------
//--------------------------------------
GreenVioletHeuristic greenVioletElement;
BlueVioletHeuristic blueVioletElement;
VioletBlueHeuristic violetBlueElement;
GreenBlueHeuristic greenBlueElement;
BlueJadeHeuristic blueJadeElement;

GreenBlobHeuristic greenBlobElement;
JadeRingHeuristic jadeRingElement;
//------------------------------------------------------
int gNumElements = 0;
BhvEelemntAbstract* gElementPtr[MAX_BONE_ELEMNTS];

int startBlue = 0;
int sizeBlue = 0;
//------------------------------------------------------
void InitBlueBoneBhv()
{
	startBlue = gNumElements;
	gElementPtr[gNumElements] = &greenVioletElement;
	gNumElements++;

	gElementPtr[gNumElements] = &blueVioletElement;
	gNumElements++;

	gElementPtr[gNumElements] = &violetBlueElement;
	gNumElements++;

	gElementPtr[gNumElements] = &greenBlueElement;
	gNumElements++;

	gElementPtr[gNumElements] = &blueJadeElement;
	gNumElements++;

	gElementPtr[gNumElements] = &greenBlobElement;
	gNumElements++;

	gElementPtr[gNumElements] = &jadeRingElement;
	gNumElements++;

	sizeBlue = gNumElements - startBlue;

}
//------------------------------------------------------

void ActiveSets::AddRedBone(BhvEelemntAbstract* bhv_ptr)
{
	redBoneSet[redSetSize].bhv_ptr = bhv_ptr;
	redBoneSet[redSetSize].num = 1;
	redSetSize++;
}

void ActiveSets::AddBlueBone(BhvEelemntAbstract* bhv_ptr)
{
	blueBoneSet[blueSetSize].bhv_ptr = bhv_ptr;
	blueBoneSet[blueSetSize].num = 1;
	blueSetSize++;
}


void ActiveSets::Pick()
{
	float red = gFiledIntegral.v[P_RED];
	float blue = gFiledIntegral.v[P_BLUE];
	float sum = red+blue;
	red /= sum;
	blue /= sum;
	statusPickRed += blue*deltaPick;
	statusPickRed = min(1, statusPickRed);
	statusPickBlue += red*deltaPick;
	statusPickBlue = min(1, statusPickBlue);
}

void GetFieldData();

void ActiveProc()
{
	GetFieldData();
	gActiveSets.Pick();
}