#ifndef _INTEGRALS_
#define _INTEGRALS_

#define MAX_SIZE 2

//------------------------------------------------------
template <int getval> class TCMax
{
public:

   float v[MAX_SIZE]; // max, min
   int ind[MAX_SIZE];

__device__	TCMax(float4* rhs, int rind)
 {
    float* rhsval = (float*)rhs;
   	v[0] = rhsval[4*rind + getval];
	v[1] = rhsval[4*rind + getval];
	ind[0] = rind;
	ind[1] = rind;
  };

__device__	TCMax(TCMax* rhs_i, int rind)
 {
	TCMax& rhs = rhs_i[rind];
	v[0] = rhs.v[0];
	v[1] = rhs.v[1];
	ind[0] = rhs.ind[0];
	ind[1] = rhs.ind[1];
  };

__device__	TCMax(int init)
 {
	v[0] = -1e10;
	v[1] = 1e10;
	ind[0] = -1;
	ind[1] = -1;
  };

__device__	TCMax& Add(float4* rhs, int rind)
 {
    float* rhsval = (float*)rhs;

	if(rhsval[4*rind + getval] > v[0])
	{
		v[0] = rhsval[4*rind + getval];
		ind[0] = rind;
	}

	if(rhsval[4*rind + getval] < v[1])
	{
		v[1] = rhsval[4*rind + getval];
		ind[1] = rind;
	}
    return *this;
  };

__device__	TCMax& Add(TCMax* rhs_i, int rind)
 {
	 TCMax& rhs = rhs_i[rind];

	if(rhs.v[0] > v[0])
	{
		v[0] = rhs.v[0];
		ind[0] = rhs.ind[0];
	}

	if(rhs.v[1] < v[1])
	{
		v[1] = rhs.v[1];
		ind[1] = rhs.ind[1];
	}
    return *this;
  };

__device__  TCMax operator+(volatile TCMax &rhs) 
  {
	  TCMax res = *this;
	  res += rhs;
	  return res;
  };

//dummy
__device__	TCMax& operator+=(volatile float4 &rhs) {
    return *this;
  };

__device__  TCMax& operator+=(volatile TCMax &rhs)
{
	if(rhs.v[0] > v[0])
	{
		v[0] = rhs.v[0];
		ind[0] = rhs.ind[0];
	}

	if(rhs.v[1] < v[1])
	{
		v[1] = rhs.v[1];
		ind[1] = rhs.ind[1];
	}
    return *this;
  };

__device__  volatile TCMax& operator =(volatile TCMax &rhs) volatile
  {
	for(int i = 0; i < MAX_SIZE; i++)
	{
		v[i] = rhs.v[i];
		ind[i] = rhs.ind[i];
	}
	return *this;
   };

};

class LSum
{
public:

  float v[8];

__device__	LSum(){};

__device__	LSum(int val)
{
	memset(v, 0, sizeof(v));
};

__device__	LSum(float4* rhs_i, int rind)
	{
		float4& rhs = rhs_i[rind];
		v[0] = fmaxf(rhs.x, 0);
		v[1] = fmaxf(-rhs.x, 0);
		v[2] = fmaxf(rhs.y, 0);
		v[3] = fmaxf(-rhs.y, 0);
		v[4] = fmaxf(rhs.z, 0);
		v[5] = fmaxf(-rhs.z, 0);
		v[6] = fmaxf(rhs.w, 0);
		v[7] = fmaxf(-rhs.w, 0);

	};

__device__	LSum(LSum* rhs_i, int rind)
	{
		LSum& rhs = rhs_i[rind];
		memcpy(v, rhs.v, sizeof(v));
	};

__device__	LSum& Add(float4* rhs_i, int rind)
	{
		float4& rhs = rhs_i[rind];
		v[0] += fmaxf(rhs.x, 0);
		v[1] += fmaxf(-rhs.x, 0);
		v[2] += fmaxf(rhs.y, 0);
		v[3] += fmaxf(-rhs.y, 0);
		v[4] += fmaxf(rhs.z, 0);
		v[5] += fmaxf(-rhs.z, 0);
		v[6] += fmaxf(rhs.w, 0);
		v[7] += fmaxf(-rhs.w, 0);
		return *this;
	};

__device__	LSum& Add(LSum* rhs_i, int rind) {
	LSum& rhs = rhs_i[rind];
	for(int i = 0; i < 8; i ++)
		v[i] += rhs.v[i];
    return *this;
  };


__device__	LSum& operator+=(volatile float4 &rhs) {
	v[0] += fmaxf(rhs.x, 0);
	v[1] += fmaxf(-rhs.x, 0);
	v[2] += fmaxf(rhs.y, 0);
	v[3] += fmaxf(-rhs.y, 0);
	v[4] += fmaxf(rhs.z, 0);
	v[5] += fmaxf(-rhs.z, 0);
	v[6] += fmaxf(rhs.w, 0);
	v[7] += fmaxf(-rhs.w, 0);
    return *this;
  };

__device__  LSum& operator+=(volatile LSum &rhs)
  {
	for(int i = 0; i < 8; i ++)
		v[i] += rhs.v[i];

    return *this;
  };

__device__  LSum operator+(volatile LSum &rhs) 
  {
	  LSum res = *this;
	  res += rhs;
	  return res;
  };

__device__  volatile LSum& operator =(volatile LSum &rhs) volatile
  {
	for(int i = 0; i < 8; i ++)
		v[i] = rhs.v[i];

    return *this;
   };
};
//----------------------------------------------------------
template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};


// specialize for double to avoid unaligned memory 
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double*()
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }

    __device__ inline operator const double*() const
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }
};

#endif
