#ifndef _NCMATH_H_
#define _NCMATH_H_
//non-cuda math

inline float4 operator+(const float4& lhs, const float4& rhs) 
{
    float4 temp = lhs;
	temp.x += rhs.x;
	temp.y += rhs.y;
	temp.z += rhs.z;
	temp.w += rhs.w;
   return temp;
}

inline float4 operator*(const float& a, const float4& b) 
{
   float4 temp = b;
   temp.x *= a;
   temp.y *= a;
   temp.z *= a;
   temp.w *= a;
   return temp;
}

inline float2 operator+(const float2& lhs, const float2& rhs) 
{
    float2 temp = lhs;
	temp.x += rhs.x;
	temp.y += rhs.y;
   return temp;
}

 inline float2 operator-(const float2& lhs, const float2& rhs) 
{
    float2 temp = lhs;
	temp.x -= rhs.x;
	temp.y -= rhs.y;
   return temp;
}

inline float2 operator*(const float& a, const float2& b) 
{
   float2 temp = b;
   temp.x *= a;
   temp.y *= a;
   return temp;
}


#endif