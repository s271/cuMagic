#ifndef _LAYER_DEFINES_
#define _LAYER_DEFINES_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define M_PI 3.14159265359f
#define I2F (1.0f / 4294967296.0f)

#define MAX_LAYERS 2

#define SWIDTH 1024
#define SHEIGHT 768

#define CTRL_WIDTH 256

extern unsigned int sim_width;
extern unsigned int sim_height;
#define sim_rect (sim_width*sim_height)

#define INTERP_SIZEX (24)
#define INTERP_SIZEY (18)
#define INTERP_SIZEZ (12)

#define MAX_PL_VALUE (.001f)
#define INV_PL_VALUE (1.f/MAX_PL_VALUE)

#define DO_DEFAULT 0
#define DO_FORM0 1
#define DO_VFILED0 2
#define DO_VFILED1 3
#define DO_VFILED2 4
#define DO_VFILED3 5

#define MIN_C_STATE 0
#define MAX_C_STATE 4

#define NFLAYERS 4

#include  <minmax.h>

#define NUM_COLORS 8

#define MAX_INTERACTIONS (NUM_COLORS+1)

#define P_BLACK 0
#define P_RED 1
#define P_BLUE 2
#define P_GREEN 3
#define P_YELLOW 4
#define P_ORANGE 5
#define P_VIOLET 6
#define P_JADE 7
#define P_MAGNETA 8


struct GSum
{
	float v[NUM_COLORS+1];
};

struct TCMaxHost
{
   float v[2]; // max, min
   int ind[2];
};


#define ME_BLUE 0
#define ME_RED 1
#define ME_ORANGE 4
#define ME_VIOLET 5
#define ME_GREEN 2
#define ME_YELLOW 3
#define ME_JADE 6
#define ME_MAGNETA 7

#define STACK_TSIZE 4

#endif