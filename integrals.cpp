#include <GL/glew.h>

// CUDA utilities and system includes
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <rendercheck_gl.h>
#include <cudaGL.h>

// Shared Library Test Functions
#include <shrUtils.h>
#include "layer_defines.h"

GSum gFiledIntegral;
extern float *gRedBlueField;
extern float* cuTempData;
extern float *gPhysLayer[MAX_LAYERS];

void GetFiledIntegrals(float* v, GSum& integral)
{
	memset(integral.v, 0, sizeof(integral.v));
	integral.v[P_RED] = v[1];
	integral.v[P_BLUE] = v[0];

	integral.v[P_GREEN] = v[2];
	integral.v[P_YELLOW] = v[3];

	integral.v[P_ORANGE] = v[4];
	integral.v[P_VIOLET] = v[5];

	integral.v[P_JADE] = v[6];
	integral.v[P_MAGNETA] = v[7];

};

class LSum;
template <class Tin, class Tout, bool isCoord>
void reduce(int size, int threads, int blocks, Tin *d_idata, Tout *d_odata);

template <int geti> class TCMax;
template void reduce<float4, TCMax<0>, true>(int size, int threads, int blocks, 
           float4 *d_idata, TCMax<0> *d_odata);


template <class Tin, class Tout>
void reduceSmallC(int xside, int yside, int blockSize, Tin *d_idata, Tout *d_odata);

template void reduceSmallC<float4, TCMax<0>>(int xside, int yside,
										  int blockSize, float4 *d_idata, TCMax<0> *d_odata);

extern float *tempHostData;

void GetFieldSums()
{
	int nBlocks = 64;
	int nThresds = 256;

	float* hdata = tempHostData;

	reduce<float4, LSum, false>(sim_width*sim_height, nThresds, nBlocks, (float4*)gRedBlueField, (LSum*)cuTempData);
    cutilSafeCallNoSync( cudaMemcpy( hdata, cuTempData, nBlocks*8*sizeof(float), cudaMemcpyDeviceToHost) );
	float fsum[8];
	memset(fsum, 0, sizeof(fsum));
	for(int i = 0; i < nBlocks; i++)
	for(int fi = 0; fi < 8; fi++)
	{
		fsum[fi] += hdata[8*i + fi];
	}
	GetFiledIntegrals(fsum, gFiledIntegral);
}



extern int globalDataPeriod;
TCMaxHost mmGrid[256*256];
TCMaxHost mmYGGrid[256*256];

int mmGridSize = 0;
int mmGridSizeX = 0;
int mmGridSizeY = 0;
int blockSizex= 64;
int blockSizey= 64;

void Reduce2DBlockH(int xs, int ys, int widthx, int widthy,
				   int xside, int& nNewBlocks,
				   float4* cuInputFiels, float* cuTempData, TCMaxHost* hostOutData);
void Reduce2DBlockYG(int xs, int ys, int widthx, int widthy,
				   int xside, int& nNewBlocks,
				   float4* cuInputFiels, float* cuTempData, TCMaxHost* hostOutData);
extern float debugArr[256*256*8];


void GetLocalMaxMin()
{
	static int dataCounter1 = 0;
	if(dataCounter1 > globalDataPeriod)
		dataCounter1 = 0;

	if(dataCounter1 == 0)
	{

		float* hdata = tempHostData;

		memset(mmGrid, 0, sizeof(mmGrid));
		memset(mmYGGrid, 0, sizeof(mmYGGrid));

		int nBlocksx = mmGridSizeX;
		int nBlocksy = mmGridSizeY;


		for(int bx = 0; bx < nBlocksx; bx++)
		for(int by = 0; by < nBlocksy; by++)
		{
			int ind = bx + by*nBlocksx;
			int nNewBlocks=-1;	

			Reduce2DBlockH(bx*blockSizex, by*blockSizey, blockSizex, blockSizey,
				   sim_width, nNewBlocks,
				   (float4*)gRedBlueField, cuTempData, &mmGrid[ind]);

			Reduce2DBlockYG(bx*blockSizex, by*blockSizey, blockSizex, blockSizey,
				   sim_width, nNewBlocks,
				   (float4*)gRedBlueField, cuTempData, &mmYGGrid[ind]);
		}

		int delta_nonmax = 64;
		float rbThreshold = .2;


		for(int bx = 0; bx < nBlocksx; bx++)
		for(int by = 0; by < nBlocksy; by++)
		{
			int ind = bx + by*nBlocksx;

			int iy0 = mmGrid[ind].ind[0]/sim_width;
			int ix0 =  mmGrid[ind].ind[0] - iy0*sim_width;

			int iy1 = mmGrid[ind].ind[1]/sim_width;
			int ix1 =  mmGrid[ind].ind[1] - iy1*sim_width;

			if(mmGrid[ind].v[0] < rbThreshold)
			{
				mmGrid[ind].ind[0] = -1;
			}

			if( mmGrid[ind].v[1] > -rbThreshold)
			{
				mmGrid[ind].ind[1] = -1;
			}
//nonmax suppression
			for(int dx = -1; dx <= 1; dx++)
			for(int dy = -1; dy <= 1; dy++)
			{
				if(dx==0&&dy==0)
					continue;

				int bx1 = bx + dx;
				int by1 = by + dy;
				if(bx1 < 0 || by1 < 0 || bx1 >= nBlocksx || by1 >= nBlocksy)
					continue;
				int ind1 = bx1 + by1*nBlocksx;

				if(mmGrid[ind1].ind[0] >= 0 && mmGrid[ind1].v[0] < mmGrid[ind].v[0])
				{

					int iyt0 = mmGrid[ind1].ind[0]/sim_width;
					int ixt0 =  mmGrid[ind1].ind[0] - iyt0*sim_width;
					
					if(abs(ix0 - ixt0) < delta_nonmax && abs(iy0 - iyt0) < delta_nonmax)
						mmGrid[ind1].ind[0] = -1;
				}

				if(mmGrid[ind1].ind[1] >= 0 && mmGrid[ind1].v[1] > mmGrid[ind].v[1])
				{
					int iyt1 = mmGrid[ind1].ind[1]/sim_width;
					int ixt1 =  mmGrid[ind1].ind[1] - iyt1*sim_width;
					
					if(abs(ix1 - ixt1) < delta_nonmax && abs(iy1 - iyt1) < delta_nonmax)
						mmGrid[ind1].ind[1] = -1;
				}
			}

		}

	}
	dataCounter1++;
}


void GetFieldData()
{
	GetFieldSums();
	GetLocalMaxMin();
};

extern float* grid8ValTick;
extern float threshold_YG;
float threshold_JM = 100;

int GetYGPos(int& resx, int& resy, int ygind)// 0 is green, 1 is yellow
{

//first pass
		for(int ipass = 0; ipass < 2; ipass++)
		{
			float vMax = 0;
			int indfound = -1;
			for(int bx = 0; bx < mmGridSizeX; bx++)
			for(int by = 0; by < mmGridSizeY; by++)
			{
				int ind = bx + by*mmGridSizeX;

				if(ipass == 0 && grid8ValTick[8*ind + ME_JADE + 1-ygind] > threshold_JM)
					continue;

				float val =(1-2*ygind)*mmYGGrid[ind].v[ygind];
				if(val > vMax)
				{
					vMax = val;
					indfound = mmYGGrid[ind].ind[ygind];
				}
			}

			if(indfound >= 0)
			{
				resx = indfound/sim_width;
				resy =  indfound - resx*sim_width;
				return 0;
			}
		}

		return -1;
}
