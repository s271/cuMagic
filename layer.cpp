#include "cutil_inline.h"
#include "layer_defines.h"
#include "math.h"
#include "glcontrols.h"
#include "bhv.h"

void InitRnd2DF(int seed, float* temp, int intsizex, int intsizey);
void InitRnd2DInt(int seed, unsigned int* temp, int intsizex, int intsizey);

void Spline2D(float* interpGrid, int intsizex, int intsizey, float* temp_grid, float scaleadd, float* out, int xsideout, int ysideout);
void InitPhysLayer(float* cuPhysLayer, unsigned int* cuStateLayer, unsigned int* cuRand, int xside, int yside);
void Randomize(unsigned int* cuRand, int imgw, int imgh);
void InitSmooth(float attf, int sim_width, float* temp, cudaArray* out);
void InitWavePack(int nf, float attf, int sim_width, int sim_height, float* temp, cudaArray* out);
void InitFuncLayer(cudaArray* cuFuncLayer, float* cuTemp, int xside, int yside);
void FieldLayerProc(float* cuLandscape, float* fieldLayerIn, float* fieldLayerOut,
					float* vectField, int xside, int yside);
void ParticleStateInit(float* cuLandscape, unsigned int* cuRand, 
					   unsigned int* cuStateOut, float* cuPhysLayerIn,
					   float* fieldLayer);
void LayerProc(int imgw, int imgh, cudaArray* arr0, cudaArray* arr1, float* res, float tx, float ty, float tx1, float ty1);
void LayerProcFunc(int imgw, int imgh, cudaArray* arr0, cudaArray* arr1, cudaArray* funcArr, float* res, float tx, float ty);

void ParticleLayerProc(float* res, unsigned int* cuRand,
					   unsigned int* cuStateIn,
					   unsigned int* cuStateOut,
					   float* particleLayerIn, 
					   float* fieldLayer,
					   float* vectField,
					   float* particleLayerOut,
					   float tx, float ty);

void Float2Int(int imgw, int imgh, float* cuTLayer, float* cuPhysLayer,
			   unsigned int* cuStateIn, float* fieldLayer,
			   float* vectField, unsigned int* cuda_int);
void initColors();

cudaArray *gCudaVectArray = NULL;
cudaArray *gCudaFlArray = NULL;
cudaArray *gCudaLayer[MAX_LAYERS];
cudaArray *gCudaFuncLayer[MAX_LAYERS];
cudaArray *gCudaFuncWavePack = NULL;
cudaArray *gCudaFuncSmooth = NULL;

float* cuTempData = NULL;
#define TEMP_SIZE 8
unsigned int* cuRandArr = NULL;
float *gRedBlueField = NULL;

float *gVectorLayer = NULL;

float *gPhysLayer[MAX_LAYERS];
unsigned int* gStateLayer[MAX_LAYERS]; 

#define TEMP_HOST_ELEM 8
float *tempHostData = NULL;
float *tempHostDataNoCuda = NULL;
float* grid8ValTick = NULL;
void InitBhv();

extern float2 stackTPos0[STACK_TSIZE];
extern float2 stackTPos1[STACK_TSIZE];
ObjInertia gObjInertia;

float gObj0X = 600, gObj0Y = 200;
float gObj1X = 400, gObj1Y = 400;

extern int mmGridSize;
extern int mmGridSizeX;
extern int mmGridSizeY;
extern int blockSizex;
extern int blockSizey;
extern TCMaxHost mmGrid[256*256];
extern TCMaxHost mmYGGrid[256*256];

void InitCudaLayers()
{

	mmGridSizeX = sim_width/blockSizex;
	mmGridSizeY = sim_height/blockSizey;
	mmGridSize = mmGridSizeX*mmGridSizeY;
	memset(mmGrid, 0, sizeof(mmGrid));
	memset(mmYGGrid, 0, sizeof(mmYGGrid));

	tempHostData = (float*)malloc(sim_width*sim_height*TEMP_HOST_ELEM*sizeof(float));
	tempHostDataNoCuda = (float*)malloc(sim_width*sim_height*TEMP_HOST_ELEM*sizeof(float));
	grid8ValTick = (float*)malloc(sim_width*sim_height*8*sizeof(float));

	initColors();

	memset(gCudaLayer, 0, sizeof(gCudaLayer));
	memset(gCudaFuncLayer, 0, sizeof(gCudaFuncLayer));
	memset(gPhysLayer, 0, sizeof(gPhysLayer));
	memset(gStateLayer, 0, sizeof(gStateLayer));

	srand(0);
	int seed = rand();

	const cudaChannelFormatDesc desc4 = cudaCreateChannelDesc<float4>();
	cudaMallocArray(&gCudaVectArray, &desc4, sim_width, sim_height);
#if NFLAYERS ==2
	const cudaChannelFormatDesc desc2 = cudaCreateChannelDesc<float2>();
#else if NFLAYERS ==4
	const cudaChannelFormatDesc descF = desc4;
#endif
	cudaMallocArray(&gCudaFlArray, &descF, sim_width, sim_height);

	const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaMallocArray(&gCudaFuncWavePack, &desc, sim_width);
	cudaMallocArray(&gCudaFuncSmooth, &desc, sim_width);

	cudaMallocArray(&(gCudaLayer[0]), &desc, sim_width, sim_height);
	cudaMallocArray(&(gCudaLayer[1]), &desc, sim_width, sim_height);

	cudaMallocArray(&(gCudaFuncLayer[0]), &desc, sim_width, sim_height);

	cudaMalloc(&cuTempData, TEMP_SIZE*sizeof(float)*sim_width*sim_height);
	cudaMalloc(&cuRandArr, sizeof(unsigned int)*sim_width*sim_height);

	cudaMalloc(&gStateLayer[0], sim_rect*sizeof(float));
	cudaMemset(gStateLayer[0], 0, sim_rect*sizeof(float));
	cudaMalloc(&gStateLayer[1], sim_rect*sizeof(float));
	cudaMemset(gStateLayer[1], 0, sim_rect*sizeof(float));

	cudaMalloc(&gPhysLayer[0], sim_rect*sizeof(float));
	cudaMemset(gPhysLayer[0], 0, sim_rect*sizeof(float));
	cudaMalloc(&gPhysLayer[1], sim_rect*sizeof(float));
	cudaMemset(gPhysLayer[1], 0, sim_rect*sizeof(float));

	cudaMalloc(&gRedBlueField, NFLAYERS*sim_rect*sizeof(float));
	cudaMemset(gRedBlueField, 0, NFLAYERS*sim_rect*sizeof(float));

	size_t pitch = 4*sim_width*sizeof(float);
	cudaMallocPitch((void**)&gVectorLayer, &pitch, 4*sim_width*sizeof(float), sim_height);

	cudaMemset2D(gVectorLayer, 4*sim_width*sizeof(float), 0, 4*sim_width*sizeof(float), sim_height);	

	InitWavePack(32, 1.f, sim_width, sim_height, cuTempData, gCudaFuncWavePack);

	InitSmooth(1, sim_width, cuTempData, gCudaFuncSmooth);

	InitRnd2DInt(seed, cuRandArr, sim_width, sim_height);

	InitFuncLayer(gCudaFuncLayer[0], cuTempData, sim_width, sim_height);

	InitPhysLayer(gPhysLayer[0], gStateLayer[0], cuRandArr, sim_width, sim_height);

	float* gridIni = cuTempData+3*sim_rect/2;
	float* halfTemp = cuTempData + sim_rect;
	float* out = cuTempData + 2*sim_rect;
	cudaMemset(out, 0, sim_rect*sizeof(float));

	seed = rand();
	int gridx = INTERP_SIZEX;
	int gridy = INTERP_SIZEX;
	InitRnd2DF(seed, gridIni, gridx, gridy);
	float scaleadd = .7f;
	Spline2D(gridIni, gridx, gridy, halfTemp, scaleadd, out, sim_width, sim_height);

	seed = rand();
	gridx = (int)(gridx*2);
	gridy = (int)(gridy*2);
	InitRnd2DF(seed, gridIni, gridx, gridy);
	scaleadd = .3f;
	Spline2D(gridIni, gridx, gridy, halfTemp, scaleadd, out, sim_width, sim_height);


	cudaMemcpyToArray(gCudaLayer[0], 0, 0, out, sizeof(float)*sim_rect, cudaMemcpyDeviceToDevice);

	cudaMemset(out, 0, sim_rect*sizeof(float));
	gridx = INTERP_SIZEX;
	gridy = INTERP_SIZEX;

	seed = rand();
	InitRnd2DF(seed, gridIni, gridx, gridy);
	scaleadd = .7f;
	Spline2D(gridIni, gridx, gridy, halfTemp, scaleadd, out, sim_width, sim_height);

	seed = rand();
	gridx = (int)(gridx*1.5);
	gridy = (int)(gridy*1.5);
	InitRnd2DF(seed, gridIni, gridx, gridy);
	scaleadd = .3f;
	Spline2D(gridIni, gridx, gridy, halfTemp, scaleadd, out, sim_width, sim_height);

	cudaMemcpyToArray(gCudaLayer[1], 0, 0, out, sizeof(float)*sim_rect, cudaMemcpyDeviceToDevice);

	float2 pos0;
	pos0.x = gObj0X;
	pos0.y = gObj0Y;

	float2 pos1;
	pos1.x = gObj1X;
	pos1.y = gObj1Y;

	gObjInertia.Init(pos0, pos1);

	LayerProc(sim_width, sim_height, gCudaLayer[0], gCudaFuncLayer[0], cuTempData, pos0.x , pos0.y, pos1.x , pos1.y);
	ParticleStateInit(cuTempData, cuRandArr, 
					   gStateLayer[0], gPhysLayer[0], gRedBlueField);

	InitBhv();

}

void DeleteCudaLayers()
{
	if(tempHostData)
		free(tempHostData);

	if(tempHostDataNoCuda)
		free(tempHostDataNoCuda);

	if(grid8ValTick)
		free(grid8ValTick);

	if(cuTempData)
		cudaFree(cuTempData);

	if(cuRandArr)
		cudaFree(cuRandArr);

	if(gCudaFuncWavePack)
		cudaFreeArray(gCudaFuncWavePack);

	if(gCudaFuncSmooth)
		cudaFreeArray(gCudaFuncSmooth);

	if(gCudaVectArray)
		cudaFreeArray(gCudaVectArray);

	if(gCudaFlArray)
		cudaFreeArray(gCudaFlArray);

	if(gVectorLayer)
			cudaFree(gVectorLayer);

	if(gRedBlueField)
		cudaFree(gRedBlueField);

	for (int k = 0; k < MAX_LAYERS; k++)
	{
		if(gCudaLayer[k])
			cudaFreeArray(gCudaLayer[k]);

		if(gCudaFuncLayer[k])
			cudaFreeArray(gCudaFuncLayer[k]);

		if(gPhysLayer[k])
			cudaFree(gPhysLayer[k]);

		if(gStateLayer[k])
			cudaFree(gStateLayer[k]);
	}
}

float moveLayerx = 0;
float moveLayery = 0;
extern int gMousex;
extern int gMousey;
extern int gAction;
int currPhLayer = 0;
extern int debShowMax;

void SetVectorField(int x, int y, int rad);
void SetSpots(unsigned int* cuda_int_dest);


void processLayer( int width, int height, unsigned int* cuda_int_dest) 
{
	if(gAction == DO_FORM0)
	{
		gObj0X = (float)gMousex;
		gObj0Y = (float)gMousey;
		gAction = DO_DEFAULT;
	}

//update random
	Randomize(cuRandArr, sim_width, sim_height);

	LayerProc(width, height, gCudaLayer[0], gCudaFuncLayer[0], cuTempData, (float)gObj0X, (float)gObj0Y, gObj1X, gObj1Y);
//physical
	int outPhLayer = (currPhLayer+1)%2;
	cudaMemset(gPhysLayer[outPhLayer], 0, sim_rect*sizeof(float));

	ParticleLayerProc(cuTempData, cuRandArr, gStateLayer[currPhLayer], gStateLayer[outPhLayer],
		gPhysLayer[currPhLayer], gRedBlueField, gVectorLayer, 
		gPhysLayer[outPhLayer], moveLayerx, moveLayery);

	FieldLayerProc(cuTempData,
		gRedBlueField, gRedBlueField, 
		gVectorLayer, sim_width, sim_height);

	Float2Int(width, height, cuTempData, gPhysLayer[outPhLayer], gStateLayer[outPhLayer],
		gRedBlueField, gVectorLayer, cuda_int_dest);

	currPhLayer = outPhLayer;

	SetVectorField(gMousex, gMousey, 20);
//debug
	if(debShowMax == 1)
		SetSpots(cuda_int_dest);

}


void vectPatch( float* vectField, float cx, float cy, float r, int type,
								float vx, float vy, int xside, int yside);

void SetVectorField(int x, int y, int rad)
{
	if(gAction < DO_VFILED0 || gAction > DO_VFILED3)
		return;

	int type = gAction - DO_VFILED0;

	float a = (float)(rand()%360);

	float vx = .5f + .5f*cos(2*M_PI*a/360);
	float vy = .5f + .5f*sin(2*M_PI*a/360);

	vectPatch(gVectorLayer, (float)x, (float)y, (float)rad, type,
					vx,  vy, sim_width, sim_height);

	gAction = DO_DEFAULT;
}

//debug
extern TCMaxHost mmGrid[];
extern int mmGridSize;

void SetSpots(unsigned int* cuda_int_dest)
{
	int kk = 0;
	for(int i=0; i < mmGridSize; i++)
	{
		if(mmGrid[i].ind[kk] >= 0 && mmGrid[i].v[kk]*(1-2*kk) >= 0)
		{
			unsigned int val = 255<<8;
			 cutilSafeCallNoSync( cudaMemcpy( cuda_int_dest + mmGrid[i].ind[kk], &val, sizeof(int), cudaMemcpyHostToDevice) );
		}
	}

	int delta = 128;
	for(unsigned int i=0; i < sim_height/delta; i++)
	{
		unsigned int val = 255;
		cutilSafeCallNoSync( cudaMemset(cuda_int_dest + i*delta*sim_width, val, sim_width*sizeof(int)) );
	}
}

void GetPoints(int xs, int ys, int xe, int ye, int size, float* arr)
{
	float dx = (xe-xs)*1.f/(size-1);
	float dy = (ye-ys)*1.f/(size-1);

	for(int i=0; i < size; i++)
	{
		int ix = (int)(xs + i*dx);
		int iy = (int)(ys + i*dy);
		cudaMemcpy(arr + 4*i, gRedBlueField + 4*(ix + iy*sim_width), 4*sizeof(float), cudaMemcpyDeviceToHost);
	}

}

void GetPoint(int ix, int iy, float* valArr)
{
		cudaMemcpy(valArr, gRedBlueField + 4*(ix + iy*sim_width), 4*sizeof(float), cudaMemcpyDeviceToHost);
}


#include "ncmath.h"

void ObjInertia::Step(float tx, float ty, float tx1, float ty1)
{
	float2 vp0_prev = vp0;
	float2 vp1_prev = vp1;

	vp0.x = tx - stackTPos0[spos0].x;
	vp0.y = ty - stackTPos0[spos0].y;

	vp1.x = tx1 - stackTPos1[spos1].x;
	vp1.y = ty1 - stackTPos1[spos1].y;

	float2 acc0 = vp0-vp0_prev;
	float2 acc1 = vp1-vp1_prev;
//calc attenuation

	float vabs0 = sqrt(vp0.x*vp0.x + vp0.y*vp0.y);
	float absacc0 = sqrt(acc0.x*acc0.x + acc0.y*acc0.y);

	static int nn=0;
	static float avgv=0;
	static float avgacc = 0;

	avgv += vabs0;
	avgacc = max(absacc0, avgacc);

	nn++;
//only my!

	atten0 = max(0, atten0 - vabs0/(30.f*100) -  absacc0/(30.f*8));

	atten0 = min(1, atten0 + 1.f/(20.f*18));


	spos0 = (spos0+1)%STACK_TSIZE;
	spos1 = (spos1+1)%STACK_TSIZE;

	stackTPos0[spos0].x = tx;
	stackTPos0[spos0].y = ty;

	stackTPos1[spos1].x = tx1;
	stackTPos1[spos1].y = ty1;
}