inline int calc_lut(__local int* smem, int val, int tid)
{
    smem[tid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid == 0)
        for (int i = 1; i < 256; ++i)
            smem[i] += smem[i - 1];
    barrier(CLK_LOCAL_MEM_FENCE);

    return smem[tid];
}

inline int reduce(__local volatile int* smem, int val, int tid)
{
    smem[tid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 128)
        smem[tid] +=smem[tid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 64)
        smem[tid] +=smem[tid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32)
    {
        smem[tid] += smem[tid + 32];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 16)
    {
        smem[tid] += smem[tid + 16];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 8)
    {
        smem[tid] += smem[tid + 8];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 4)
    {
        smem[tid] += smem[tid + 4];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid == 0)
    {
        smem[0] = (smem[0] + smem[1]) + (smem[2] + smem[3]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    val = smem[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return val;
}


__kernel void calLut(__global __const uchar * src, const int srcStep,
                      __global uchar * lut,
                      const int2 tileSize, const int2 tiles,
                      const int clipLimit)
{
    __local int smem[512];
    int tx = get_group_id(0);
    int ty = get_group_id(1);
    int tid = get_local_id(1) * get_local_size(0)
                             + get_local_id(0);
    smem[tid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = get_local_id(1); i < tileSize.y; i += get_local_size(1))
    {
        __global const uchar* srcPtr = src + mad24(ty * tileSize.y + i, srcStep, tx * tileSize.x );
        for (int j = get_local_id(0); j < tileSize.x; j += get_local_size(0))
        {
            const int data = srcPtr[j];
            atomic_inc(&smem[data]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int tHistVal = smem[tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    int clipped = 0;
    if (tHistVal > clipLimit)
    {
        clipped = tHistVal - clipLimit;
        tHistVal = clipLimit;
    }

	clipped = reduce(smem, clipped, tid);
		
    int redistBatch = clipped / 256;
    tHistVal += redistBatch;
    int residual = clipped - redistBatch * 256;
	if(tid<residual)	
		++tHistVal;
    const int lutVal = calc_lut(smem, tHistVal, tid);
	int area=tileSize.x*tileSize.y; 
	const float scale=convert_float(255.0/area);	
    uint ires = (uint)convert_int_rte(scale*lutVal);
	lut[(ty * tiles.x + tx) * 256 + tid] =
        convert_uchar(clamp(ires, (uint)0, (uint)255));
}

__kernel void mapLut(__global __const uchar * src, const int srcStep, 
                        __global uchar * dst,__global uchar * lut,  
                        const int cols, const int rows,
                        const int2 tileSize,
                        const int2 tiles)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
	const int lutStep=256; 

    if (x >= cols || y >= rows)
        return;

    const float tyf = (convert_float(y) / tileSize.y) - 0.5f;
    int ty1 = convert_int_rtn(tyf);
    int ty2 = ty1 + 1;
    const float ya = tyf - ty1;
    ty1 = max(ty1, 0);
    ty2 = min(ty2, tiles.y - 1);

    const float txf = (convert_float(x) / tileSize.x) - 0.5f;
    int tx1 = convert_int_rtn(txf);
    int tx2 = tx1 + 1;
    const float xa = txf - tx1;
    tx1 = max(tx1, 0);
    tx2 = min(tx2, tiles.x - 1);

    const int srcVal = src[mad24(y, srcStep, x )];

    float res = 0;
    res += lut[mad24(ty1 * tiles.x+ tx1, lutStep, srcVal)] * ((1.0f - xa) * (1.0f - ya));
    res += lut[mad24(ty1 * tiles.x + tx2, lutStep, srcVal)] * ((xa) * (1.0f - ya));
    res += lut[mad24(ty2 * tiles.x + tx1, lutStep, srcVal)] * ((1.0f - xa) * (ya));
    res += lut[mad24(ty2 * tiles.x + tx2, lutStep, srcVal)] * ((xa) * (ya));


    uint ires = (uint)convert_int_rte(res);
	int idx=mad24(y, srcStep, x );
	int s_idx=mad24(cols,rows,idx); 
	int h_idx=mad24(cols,rows,s_idx); 
    dst[mad24(y, srcStep, x )] = convert_uchar(clamp(ires, (uint)0, (uint)255));
	dst[s_idx]=src[s_idx];
	dst[h_idx]=src[h_idx];
	
}

__kernel void rgb2hsv(__global __const uchar * src, const int srcStep, __global uchar * dst,const int cols, const int rows)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
   
    if (x >= cols || y >= rows)
		return; 
	int idx=mad24(y,srcStep*3,x*3); 
	float r =convert_float(src[idx])/255.0;
	float g=convert_float(src[idx+1])/255.0;
    float b=convert_float(src[idx+2])/255.0;

	float3 rgb=(float3)(r,g,b); 
	float4 K = (float4)(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    float4 p = mix((float4)(rgb.zy, K.wz),(float4)(rgb.yz, K.xy), step(rgb.z, rgb.y));
    float4 q = mix((float4)(p.xyw, rgb.x),(float4)(rgb.x, p.yzx), step(p.x, rgb.x));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    float3 hsv=(float3)(fabs(q.z+(q.w-q.y)/(6.0 * d + e)),d/(q.x + e),q.x);
	uchar3 HSV = convert_uchar3_sat(hsv*255);

	uint frameSize=mad24(cols,rows,0); 
	uint v_idx=mad24(y,srcStep,x); ; 
	uint s_idx=frameSize+v_idx; 
	uint h_idx=frameSize+s_idx; 

	dst[h_idx]=HSV.x; 
	dst[s_idx]=HSV.y; 
	dst[v_idx]=HSV.z;  

}

__kernel void hsv2rgb(__global uchar* src, const int srcStep,__global uchar* dst,const int cols, const int rows )
{
	const int tidx=get_global_id(0);
	const int tidy=get_global_id(1); 
	
	if(tidx>=cols || tidy>=rows)
		return; 
	
	int frameSize=mad24(rows,cols,0); 
	int v_idx=mad24(tidy,srcStep,tidx); 
	int s_idx=frameSize+v_idx; 
	int h_idx=frameSize+s_idx; 
	
	  
    float h =convert_float(src[h_idx])/255.0;
	float s =convert_float(src[s_idx])/255.0;
	float v=convert_float(src[v_idx])/255.0;
	float3 hsv=(float3)(h,s,v);
    float4 K = (float4)(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
	float3 t=(float3)(hsv.xxx + K.xyz); 
    //float3 p = abs(fract(hsv.xxx + K.xyz) * 6.0 - K.www);
	float3 p =fabs((t-trunc(t))*6.0 - K.www);
	
    float3 rgb=hsv.z * mix(K.xxx, clamp(p-K.xxx, 0.0, 1.0), hsv.y);
	uchar3 RGB = convert_uchar3_sat(rgb*255);
	
	int idx=mad24(tidy,srcStep*3,tidx*3); 
	dst[idx]=RGB.x; 
	dst[idx+1]=RGB.y;
	dst[idx+2]=RGB.z; 
		 
}
	
