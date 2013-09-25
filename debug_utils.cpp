#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <shrUtils.h>
#include "layer_defines.h"

class LSumDeb
{
public:

  float v[8];

LSumDeb(){};

LSumDeb(float4* rhs, int ind)
	{
		v[0] = max(rhs[ind].x, 0);
		v[1] = max(-rhs[ind].x, 0);
		v[2] = max(rhs[ind].y, 0);
		v[3] = max(-rhs[ind].y, 0);
		v[4] = max(rhs[ind].z, 0);
		v[5] = max(-rhs[ind].z, 0);
		v[6] = max(rhs[ind].w, 0);
		v[7] = max(-rhs[ind].w, 0);

	};

LSumDeb& Add(float4* rhs, int rind)
	{//dummy
		*this += rhs[rind];
		return *this;
	};


LSumDeb& operator+=( float4 &rhs) {
	v[0] += max(rhs.x, 0);
	v[1] += max(-rhs.x, 0);
	v[2] += max(rhs.y, 0);
	v[3] += max(-rhs.y, 0);
	v[4] += max(rhs.z, 0);
	v[5] += max(-rhs.z, 0);
	v[6] += max(rhs.w, 0);
	v[7] += max(-rhs.w, 0);
    return *this;
  };

LSumDeb& operator+=( LSumDeb &rhs)
  {
	for(int i = 0; i < 8; i ++)
		v[i] += rhs.v[i];

    return *this;
  };

LSumDeb operator+( LSumDeb &rhs) 
  {
	  LSumDeb res = *this;
	  res += rhs;
	  return res;
  };

LSumDeb& operator =( LSumDeb &rhs) 
  {
	for(int i = 0; i < 8; i ++)
		v[i] = rhs.v[i];

    return *this;
   };
};

#define cellSizex (blockSize*2)

 void reduceCKernel(int tx, int ty, int bx, int by, int blockSize, int gridDimx,
	 LSumDeb *sdata, float4 *g_idata, LSumDeb *g_odata, int xs, int ys, unsigned int xside, unsigned int grid_dimx)
{

    unsigned int gridSize = blockSize*2*gridDimx;

#define cellSizex (blockSize*2)
	unsigned int ix = xs + bx*cellSizex + tx;
	unsigned int iy = ys + by*blockSize + ty;

	unsigned int ind = ix + xside*iy;
	unsigned int tind = tx + blockSize*ty;


    LSumDeb mySum(g_idata, ind);
	mySum.Add(g_idata, ind+blockSize);

    // each thread puts its local sum into shared memory 
    sdata[tind] = mySum;

 //   __syncthreads();


    LSumDeb* smem = sdata;
	if(tx < 16)
	{
		if (blockSize >=  32) { smem[tind] = mySum = mySum + smem[tind + 16]; }
		if (blockSize >=  16) { smem[tind] = mySum = mySum + smem[tind +  8]; }
		if (blockSize >=   8) { smem[tind] = mySum = mySum + smem[tind +  4]; }
		if (blockSize >=   4) { smem[tind] = mySum = mySum + smem[tind +  2]; }
		if (blockSize >=   2) { smem[tind] = mySum = mySum + smem[tind +  1]; }
	}
//	__syncthreads();

	unsigned int tiy = blockSize*ty;

	if(ty < 16)
	{
		if (blockSize >=  32) { smem[tiy] = mySum = mySum + smem[tiy + 16*blockSize];}//  __syncthreads();}
		if (blockSize >=  16) { smem[tiy] = mySum = mySum + smem[tiy +  8*blockSize];}//  __syncthreads();}
		if (blockSize >=   8) { smem[tiy] = mySum = mySum + smem[tiy +  4*blockSize];}//  __syncthreads();}
		if (blockSize >=   4) { smem[tiy] = mySum = mySum + smem[tiy +  2*blockSize];}//  __syncthreads();}
		if (blockSize >=   2) { smem[tiy] = mySum = mySum + smem[tiy +  1*blockSize];}//  __syncthreads();}
	}

	if(tind == 0)
		g_odata[bx + by*grid_dimx] = sdata[0];

}



extern float *tempHostData;
void localSumDeb(int xc, int yc, int width, int xside, int yside, float4 *field, LSumDeb* cuTempData, float* fsum)
{
	int xs = max(0, xc - width/2);
	int ys = max(0, yc - width/2);

	int xe = min(xc + width/2, xside-1);
	int ye = min(yc + width/2, yside-1);

	width = min(xe-xs, ye-ys);

	width -= width%64;
	xs = xc - width/2;
	ys = yc - width/2;


	int blockSize = 32;
	int blocksx = width/cellSizex;
	int blocksy = width/blockSize; 

    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid(blocksx, blocksy, 1);
	int smemSize = blockSize*blockSize* sizeof(LSumDeb);

 //   reduceCKernel<float4, LSum, 32><<< dimGrid, dimBlock, smemSize >>>(field, cuTempData, xs, ys, xside, blocksx);
	float* hdata = tempHostData;
	int nBlocks = blocksx*blocksy;
	//cudaMemcpy(hdata, cuTempData, nBlocks*sizeof(LSum), cudaMemcpyDeviceToHost);
	memset(fsum, 0, 8*sizeof(float));
	for(int i = 0; i < nBlocks; i++)
	for(int fi = 0; fi < 8; fi++)
	{
		fsum[fi] += hdata[8*i + fi];
	}

}
////-------------------
//	int xs = max(0, xc - width/2);
//	int ys = max(0, yc - width/2);
//	int xside = sim_width;
//	int yside = sim_height;
//
//	int xe = min(xc + width/2, xside-1);
//	int ye = min(yc + width/2, yside-1);
//
//	width = xe-xs;
//	width -= width%64;
//
//	xs = xc - width/2;
//	ys = yc - width/2;
//
//	xe = xs + width;
//	ye = ys + width;
//
//#define cellSizex (blockSize*2)
//
//	int blockSize = 32;
//	int blocksx = width/cellSizex;
//
//	int wx = blocksx*32*2;
//
//	for(int iy =  ys; iy < ye; iy++)
//		cudaMemcpy(debugArr + 4*(iy-ys)*wx, gRedBlueField + 4*(xs + iy*sim_width), 4*wx*sizeof(float), cudaMemcpyDeviceToHost);
//
//	float sum[4];
//	float pr =0;
//	float pb = 0;
//	memset(sum, 0, sizeof(sum));
//	for(int iy =  0; iy < ye-ys; iy++)
//	for(int ix =  0; ix < blocksx*32*2; ix++)
//	{
//		for(int k =0; k < 4; k++)
//			sum[k] += debugArr[4*(ix + iy*wx)+k];
//		if(debugArr[4*(ix + iy*wx)] > 0)
//			pb += debugArr[4*(ix + iy*wx)];
//		else
//			pr += debugArr[4*(ix + iy*wx)];
//	}
//
////------------------------


//-------------------
	//int xs = bx*blockSizex;
	//int ys = by*blockSizey;
	//int xside = sim_width;
	//int yside = sim_height;
	//int width = 64;

	//int xe = xs + width;
	//int ye = ys + width;


	//int wx = blockSizex;

	//for(int iy =  ys; iy < ye; iy++)
	//	cudaMemcpy(debugArr + 4*(iy-ys)*wx, gRedBlueField + 4*(xs + iy*sim_width), 4*wx*sizeof(float), cudaMemcpyDeviceToHost);

	//float sum[4];
	//float prmax = -10000000;
	//float prmin = 10000000;
	//float pbmax = -10000000;
	//float pbmin = 10000000;	
	//for(int iy =  0; iy < ye-ys; iy++)
	//for(int ix =  0; ix < 64; ix++)
	//{

	//	if(debugArr[4*(ix + iy*wx)] > 0)
	//	{
	//		prmax = max(debugArr[4*(ix + iy*wx)], prmax);
	//		prmin = min(debugArr[4*(ix + iy*wx)], prmin);
	//		
	//	}
	//	else
	//	{
	//		pbmax = max(debugArr[4*(ix + iy*wx)], pbmax);
	//		pbmin = min(debugArr[4*(ix + iy*wx)], pbmin);
	//	}
	//}

//------------------------