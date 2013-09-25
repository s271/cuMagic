#ifndef _GL_CONTROLS_
#define _GL_CONTROLS_
#include "layer_defines.h"


class vector2
{
public:
	float x, y;
	vector2(float xin, float yin)
	{
		x = xin;
		y = yin;
	}

	vector2(){};
};

inline vector2 operator+(const vector2& lhs, const vector2& rhs) 
{
    vector2 temp = lhs;
	temp.x += rhs.x;
	temp.y += rhs.y;
   return temp;
}

inline vector2 operator-(const vector2& lhs, const vector2& rhs) 
{
    vector2 temp = lhs;
	temp.x -= rhs.x;
	temp.y -= rhs.y;
   return temp;
}

inline vector2 operator*(const float& a, const vector2& b) 
{
   vector2 temp = b;
   temp.x *= a;
   temp.y *= a;
   return temp;
}

#define CIRCL_SEG 16

struct GeomStorage
{
	vector2 star6[6];
	vector2 circle[CIRCL_SEG];
};
/*
1.0, 0.0, 0.0,
.8, .4, .08,
.63f, .24f, .86f,
0.0, 0.0, 1.0,
0, .8, 0.0,
1, 1, 0,
.2, .2, .2,
*/

#define RED 0
#define YELLOW 1
#define VIOLET 2
#define BLUE 3
#define GREEN 4
#define ORANGE 5
#define MAGNETA 6
#define JADE 7


#define BLACK (MAX_INTERACTIONS-1)
#define PMAX (MAX_INTERACTIONS+1)

class CtrlsSinglet
{
public:
	vector2 bigHexCnt;
	vector2 bigHex[MAX_INTERACTIONS];
	vector2 minihex[MAX_INTERACTIONS][6];
	int interactMatrixF[MAX_INTERACTIONS][MAX_INTERACTIONS];
	int interactMatrixP[MAX_INTERACTIONS][MAX_INTERACTIONS];
	void initCtrls();
};

void initGeom();

extern CtrlsSinglet controls;
extern int C2P[];
extern int P2C[];
#endif