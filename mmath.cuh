#include <cutil_inline.h> 

#define M_PI 3.14159265359f

__device__ inline float fracf(float x)
{
	return x - floorf(x);
}
 //(jsr^=(jsr<<17), jsr^=(jsr>>13), jsr^=(jsr<<5))
__device__ inline void rand_xorshift(unsigned int& rng_state)
{
    // Xorshift algorithm from George Marsaglia's paper
    rng_state ^= (rng_state << 17);
    rng_state ^= (rng_state >> 13);
    rng_state ^= (rng_state << 5);
}

__device__ inline float clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

__device__ inline int clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}

#define XMASK 1023


#define WARPX(x) ((x+SWIDTH)&XMASK)
//#define WARPX(x) ((x+SWIDTH)%SWIDTH)
#define WARPY(y) ((y+SHEIGHT)%SHEIGHT)
#define DELTA_M 0

__device__ inline int signf(float a)
{
	return (a == 0)?0:(1-2*signbit(a));
}


#define SHARED_MEM(x, y) \
    SMEM(RSH + tx, RSH + ty) = getVal(x, y);\
    if (tx < RSH) {\
        SMEM(tx, RSH + ty) = getVal(x - RSH, y);\
        SMEM(RSH + bw + tx, RSH + ty) = getVal(x + bw, y);\
    }\
    if (ty < RSH) {\
        SMEM(RSH + tx, ty) = getVal(x, y - RSH);\
        SMEM(RSH + tx, RSH + bh + ty) = getVal(x, y + bh);\
    }\
    if ((tx < RSH) && (ty < RSH)) {\
        SMEM(tx, ty) = getVal(x - RSH, y - RSH);\
        SMEM(tx, RSH + bh + ty) = getVal(x - RSH, y + bh);\
        SMEM(RSH + bw + tx, ty) = getVal(x + bh, y - RSH);\
        SMEM(RSH + bw + tx, RSH + bh + ty) = getVal(x + bw, y + bh);\
    }

#define COPY_SHARED_MEM(x, y) \
    SCOPY(RSH + tx, RSH + ty, x, y);\
    if (tx < RSH) {\
        SCOPY(tx, RSH + ty, x - RSH, y);\
        SCOPY(RSH + bw + tx, RSH + ty, x + bw, y);\
    }\
    if (ty < RSH) {\
        SCOPY(RSH + tx, ty, x, y - RSH);\
        SCOPY(RSH + tx, RSH + bh + ty, x, y + bh);\
    }\
    if ((tx < RSH) && (ty < RSH)) {\
        SCOPY(tx, ty, x - RSH, y - RSH);\
        SCOPY(tx, RSH + bh + ty, x - RSH, y + bh);\
        SCOPY(RSH + bw + tx, ty, x + bh, y - RSH);\
        SCOPY(RSH + bw + tx, RSH + bh + ty, x + bw, y + bh);\
    }

__device__  inline float4 operator+(const float4& lhs, const float4& rhs) 
{
    float4 temp = lhs;
	temp.x += rhs.x;
	temp.y += rhs.y;
	temp.z += rhs.z;
	temp.w += rhs.w;
   return temp;
}

__device__  inline float4 operator*(const float& a, const float4& b) 
{
   float4 temp = b;
   temp.x *= a;
   temp.y *= a;
   temp.z *= a;
   temp.w *= a;
   return temp;
}

__device__  inline float2 operator+(const float2& lhs, const float2& rhs) 
{
    float2 temp = lhs;
	temp.x += rhs.x;
	temp.y += rhs.y;
   return temp;
}

__device__  inline float2 operator-(const float2& lhs, const float2& rhs) 
{
    float2 temp = lhs;
	temp.x -= rhs.x;
	temp.y -= rhs.y;
   return temp;
}

__device__  inline float2 operator*(const float& a, const float2& b) 
{
   float2 temp = b;
   temp.x *= a;
   temp.y *= a;
   return temp;
}