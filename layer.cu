#include <shrUtils.h>
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include "mmath.cuh"
#include "layer_defines.h"
#include "bhv.h"

texture<float, 2, cudaReadModeElementType> inTex0;
texture<float, 2, cudaReadModeElementType> inTex1;
texture<float, 1, cudaReadModeElementType> funcTex;

__global__ void cudaProcessKernel(float* res, float ttx, float tty,  float ttx1, float tty1,
								  float invx, float invy, float sina, float cosa,
								  float atten0, float atten1, int xside, int yside)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int ix = blockIdx.x*bw + tx;
    int iy = blockIdx.y*bh + ty;

	if(ix >= xside || iy >= yside)
		return;

	float cx = xside*.5f;
	float cy = yside*.5f;

	int ind = ix + xside*iy;

	float fx =(ix-ttx);
	float fy =(iy-tty);
	float fxt = invx*fminf(fmaxf(sina*fx + cosa*fy+cx, 0), yside);
	float fyt = invy*fminf(fmaxf(cosa*fx - sina*fy+cy, 0), yside);

	float fx1 =(ix-ttx1);
	float fy1 =(iy-tty1);
	float fxt1 = invx*fminf(fmaxf(sina*fx1 + cosa*fy1+cx, 0), yside);
	float fyt1 = invy*fminf(fmaxf(cosa*fx1 - sina*fy1+cy, 0), yside);


	float val = .5*tex2D(inTex0, invx*(ix), invy*(iy));
	val += atten0*tex2D(inTex1, fxt, fyt);
	val += atten1*tex2D(inTex1, fxt1, fyt1);
	float w = (1.f/2);
	res[ind] = fminf(w*val, 1);

}
float angle = 0;
float dangle = M_PI*2.f/180;

extern ObjInertia gObjInertia;

void LayerProc(int imgw, int imgh, cudaArray* arr0, cudaArray* arr1, float* res, float tx, float ty, float tx1, float ty1)
{
    struct cudaChannelFormatDesc desc; 
	
    cutilSafeCall(cudaGetChannelDesc(&desc, arr0));

	inTex0.addressMode[0] = cudaAddressModeWrap;
	inTex0.addressMode[1] = cudaAddressModeWrap;
	inTex0.filterMode = cudaFilterModeLinear;
	inTex0.normalized = true;

	inTex1.addressMode[0] = cudaAddressModeWrap;
	inTex1.addressMode[1] = cudaAddressModeWrap;
	inTex1.filterMode = cudaFilterModeLinear;
	inTex1.normalized = true;

    cutilSafeCall(cudaBindTextureToArray(&inTex0, arr0, &desc));
	cutilSafeCall(cudaBindTextureToArray(&inTex1, arr1, &desc));
 
	int blockx = 32;
	int blocky = 32;

    dim3 grid((imgw/blockx)+(!(imgw%blockx)?0:1), (imgh/blocky)+(!(imgh%blocky)?0:1));
    dim3 block(blockx, blocky);    

	angle = fmodf(angle-dangle, 2*M_PI);
	float sina = sin(angle);
	float cosa = cos(angle);

	cudaFuncSetCacheConfig(cudaProcessKernel, cudaFuncCachePreferL1);

	gObjInertia.Step(tx, ty, tx1, ty1);

    cudaProcessKernel<<< grid, block >>> (res, tx, ty, tx1, ty1, 1.f/imgw, 1.f/imgh,
		sina, cosa, gObjInertia.atten0, gObjInertia.atten1, imgw, imgh);

	cudaUnbindTexture(inTex0);
	cudaUnbindTexture(inTex1);
}

//------------------------------------------------------------------------
__device__ float cuPColor[MAX_INTERACTIONS*4];


__device__ inline int rgbToInt(float r, float g, float b)
{
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b)<<16) | (int(g)<<8) | int(r);
}

__global__ void Float2IntKernel(const float* cuTLayer, const float* cuPhysLayer,  unsigned int* cuState,
								const float* cuFieldLayer, float* vectField,
								unsigned int* cuda_int, int xside, int yside)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int ix = blockIdx.x*bw + tx;
    int iy = blockIdx.y*bh + ty;

	if(ix >= xside || iy >= yside)
		return;

	int ind = ix + xside*iy;

	float tval = cuTLayer[ind];
	float val = cuPhysLayer[ind];
	int color = (cuState[ind]&15);

	float pval = val*INV_PL_VALUE;

	float fl = 1.5f*cuFieldLayer[NFLAYERS*ind];
	float filed_red = fminf(fmaxf(-fl, 0), .8f);
	float filed_blue = fminf(fmaxf(fl, 0), .8f);

	float flf = 8*cuFieldLayer[NFLAYERS*ind + 2];
	float field_orange = fminf(fmaxf(flf, 0), .8f);
	float field_violet = fminf(fmaxf(-flf, 0), .8f);

	float flmj = 4*cuFieldLayer[NFLAYERS*ind + 3];
	float field_jade = fminf(fmaxf(flmj, 0), .9f);
	float field_magneta = fminf(fmaxf(-flmj, 0), .9f);

	float fs = 3*cuFieldLayer[NFLAYERS*ind+1];
	float filed_green = fminf(fmaxf(fs, 0), .9f);
	float filed_yellow = fminf(fmaxf(-fs, 0), .9f);

	float vf = .8f*vectField[4*ind+2];
	float vw =  vectField[4*ind+3];
	float vfield = fminf(fmaxf(fabsf(vf), 0), .4f);

	float r, g, b;

	float pr = pval*cuPColor[4*color];
	float pg = pval*cuPColor[4*color+1];
	float pb = pval*cuPColor[4*color+2];

	r = tval*(1-pr) + pr;
	g = tval*(1-pg) + pg;
	b = tval*(1-pb) + pb;
//-------------------------------------------

	if(field_magneta)
	{
		float inv = 1-field_magneta;
		r = inv*r + field_magneta*.8f;
		g = inv*g + field_magneta*.1f;
		b = inv*b + field_magneta*.8f;
	}

	if(field_jade)
	{
		float inv = 1-field_jade;
		g = inv*g + field_jade*.6f;
		b = inv*b + field_jade*.5f;
	}
//---
	if(vw >0)
	{
		if(vf >= 0)
		{//green
			float inv = 1-vfield;
			r = inv*r + vfield*.1f;
			g = inv*g + vfield*.8f;
			b = inv*b + vfield*.1f;

		}
		else
		{//yellow
			float inv = 1-vfield;
			r = inv*r + vfield*1.f;
			g = inv*g + vfield*.9f;
		}
	}
	else
	{
		if(vf < 0)
		{//magneta
			float inv = 1-vfield;
			r = inv*r + vfield*.8f;
			g = inv*g + vfield*.1f;
			b = inv*b + vfield*.8f;	
		}
		else
		{//jade
			float inv = 1-vfield;
			g = inv*g + vfield*.6f;
			b = inv*b + vfield*.5f;			
		}
	}

	if(filed_blue)
	{
		float inv = 1-filed_blue;
		r = inv*r + filed_blue*.1f;
		g = inv*g + filed_blue*.2f;
		b = inv*b + filed_blue*.9f;
	}

	if(filed_red)
	{
		float inv = 1-filed_red;
		r = inv*r + filed_red*.9f;
		g = inv*g + filed_red*.2f;
		b = inv*b + filed_red*.1f;
	}

	if(field_orange)
	{
		float inv = 1-field_orange;
		r = inv*r + field_orange*.7f;
		g = inv*g + field_orange*.3f;
		b = inv*b + field_orange*.055f;
	}

	if(field_violet)
	{
		float inv = 1-field_violet;
		r = inv*r + field_violet*.5f;
		g = inv*g + field_violet*.2f;
		b = inv*b + field_violet*.7f;
	}

	if(filed_green)
	{
		float inv = 1-filed_green;
		b = inv*b + filed_green*.2f;
		g = inv*g + filed_green*.7f;
	}

	if(filed_yellow)
	{
		float inv = 1-filed_yellow;
		g = inv*g + filed_yellow*.8f;
		r = inv*r + filed_yellow*.8f;
	}

	cuda_int[ind] =  rgbToInt(255*r, 255*g, 255*b);
}


void Float2Int(int imgw, int imgh, float* cuTLayer, float* cuPhysLayer, unsigned int* cuState,
			   float* cuFieldLayer, float* vectField, unsigned int* cuda_int)
{
	int blockx = 32;
	int blocky = 32;

    dim3 grid((imgw/blockx)+(!(imgw%blockx)?0:1), (imgh/blocky)+(!(imgh%blocky)?0:1));
    dim3 block(blockx, blocky);

	cudaFuncSetCacheConfig(Float2IntKernel, cudaFuncCachePreferL1);

	Float2IntKernel<<< grid, block >>> (cuTLayer, cuPhysLayer, cuState, cuFieldLayer, vectField, cuda_int, imgw, imgh);

};


__global__ void cudaProcessKernelFunc(float* res, float ttx, float tty, float invx, float invy, int xside, int yside)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int ix = blockIdx.x*bw + tx;
    int iy = blockIdx.y*bh + ty;

	if(ix >= xside || iy >= yside)
		return;

	int ind = ix + xside*iy;

	float fny = invx/invy;

	float normx_r = invx*(ix+ttx);
	//normx_r = fmodf(fminf(fabsf(normx_r), fabsf(normx_r+1)),1);
	normx_r = fmodf(fminf(fabsf(normx_r), fabsf(normx_r+1)),1);
	float normy_r = invx*(iy+tty);
	normy_r = fmodf(fminf(fabsf(normy_r), fabsf(normy_r+fny)),fny);

	float valf = fminf(normx_r*normx_r + normy_r*normy_r, .1);

	res[ind] = .5f + .5f*(.3*tex1D(funcTex, valf) - (tex2D(inTex0, invx*(ix), invy*(iy))-.5f));

}

void LayerProcFunc(int imgw, int imgh, cudaArray* arr0, cudaArray* arr1, cudaArray* funcArr, float* res, float tx, float ty)
{
    struct cudaChannelFormatDesc desc; 	
    cutilSafeCall(cudaGetChannelDesc(&desc, arr0));

	inTex0.addressMode[0] = cudaAddressModeWrap;
	inTex0.addressMode[1] = cudaAddressModeWrap;
	inTex0.filterMode = cudaFilterModeLinear;
	inTex0.normalized = true;

	inTex1.addressMode[0] = cudaAddressModeWrap;
	inTex1.addressMode[1] = cudaAddressModeWrap;
	inTex1.filterMode = cudaFilterModeLinear;
	inTex1.normalized = true;

    cutilSafeCall(cudaBindTextureToArray(&inTex0, arr0, &desc));
	cutilSafeCall(cudaBindTextureToArray(&inTex1, arr1, &desc));
 
    struct cudaChannelFormatDesc descFunc; 
    cutilSafeCall(cudaGetChannelDesc(&descFunc, funcArr));

	funcTex.addressMode[0] = cudaAddressModeClamp;
	funcTex.filterMode = cudaFilterModeLinear;
	funcTex.normalized = true;

	cutilSafeCall(cudaBindTextureToArray(&funcTex, funcArr, &descFunc));

	int blockx = 32;
	int blocky = 8;

    dim3 grid((imgw/blockx)+(!(imgw%blockx)?0:1), (imgh/blocky)+(!(imgh%blocky)?0:1));
    dim3 block(blockx, blocky);     

    cudaProcessKernelFunc<<< grid, block >>> (res, tx, ty, 1.f/imgw, 1.f/imgh, imgw, imgh);

    cudaUnbindTexture(inTex0);
	cudaUnbindTexture(inTex1);
	cudaUnbindTexture(funcTex);


}

//------------------------------------------------------------------------

__global__ void vectPatchDiscKernel(int xs, int xe, int ys, int ye, float* vectField,
								float cx, float cy, float r2, int type0, int type1,
								float vx_, float vy_, int xside)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int ix = xs + blockIdx.x*bw + tx;
    int iy = ys + blockIdx.y*bh + ty;

	if(ix >= xe || iy >= ye)
		return;
	float lx = ix-cx;
	float ly = iy -cy;
	float rl = lx*lx + ly*ly;
	if(rl >= r2)
		return;

	float vx = lx*rsqrtf(rl);
	float vy = ly*rsqrtf(rl);

	vectField[4*ix + 4*xside*iy] = vx;
	vectField[4*ix + 1 + 4*xside*iy] = vy;
	vectField[4*ix + 2 + 4*xside*iy] = type0;
	vectField[4*ix + 3 + 4*xside*iy] = type1;

}

__global__ void vectPatchRingKernel(int xs, int xe, int ys, int ye, float* vectField,
								float cx, float cy, float r20, float r21, int type0, int type1,
								float vx_, float vy_, int xside)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int ix = xs + blockIdx.x*bw + tx;
    int iy = ys + blockIdx.y*bh + ty;

	if(ix >= xe || iy >= ye)
		return;
	float lx = ix-cx;
	float ly = iy -cy;
	float rl = lx*lx + ly*ly;
	if(rl > r21 || rl < r20)
		return;

	float vx0 = lx*rsqrtf(rl);
	float vy0 = ly*rsqrtf(rl);

	float vx = type0*vy0;
	float vy = -type0*vx0;

	vectField[4*ix + 4*xside*iy] = vx;
	vectField[4*ix + 1 + 4*xside*iy] = vy;
	vectField[4*ix + 2 + 4*xside*iy] = type0;
	vectField[4*ix + 3 + 4*xside*iy] = type1;

}

void vectPatch( float* vectField, float cx, float cy, float r, int type_list,
								float vx, float vy, int xside, int yside)
{
	int type0 = 1 - 2*(type_list%2);
	int type1 = 1 - 2*(type_list/2);

	int blockx = 32;
	int blocky = 32;

	float r2 = r*r;
	float r20 = r*r;
	float r21 = 4*r*r;

	if(type1 < 0)
		r = r21;

	int xs = max(cx - r, 0);
	int xe = min(cx + r, xside);
	int ys = max(cy - r, 0);
	int ye = min(cy + r, yside);

	int nBloclsx = ((xe-xs)/blockx)+(!((xe-xs)%blockx)?0:1);
	int nBloclsy = ((ye-ys)/blocky)+(!((ye-ys)%blocky)?0:1);

    dim3 grid(nBloclsx, nBloclsy);
    dim3 block(blockx, blocky);

	if(type1 >= 0)
		vectPatchDiscKernel<<< grid, block >>>(xs, xe, ys, ye, vectField,
									cx, cy, r2, type0, type1,
									vx, vy, xside);
	else
		vectPatchRingKernel<<< grid, block >>>(xs, xe, ys, ye, vectField,
									cx, cy, r20, r21, type0, type1,
									vx, vy, xside);


};

#include "glcontrols.h"
extern float ColorsHex[];
float P2Color[MAX_INTERACTIONS*4];
void initColors()
{
//init p2colors
	memset(P2Color, 0, sizeof(P2Color));
	for(int i = 1; i < MAX_INTERACTIONS; i++)
	{
		for(int kc = 0; kc < 3; kc++)
			P2Color[4*i + kc] = ColorsHex[3*P2C[i] + kc];
	}

	cudaMemcpyToSymbol(cuPColor, P2Color, sizeof(P2Color), 0, cudaMemcpyHostToDevice);
}