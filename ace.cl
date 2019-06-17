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

	//printf("tid:%d,tHistV%d\t",tid,tHistVal); 
    const int lutVal = calc_lut(smem, tHistVal, tid);

	int area=tileSize.x*tileSize.y; 
	float scale=255.0/area; 
    uint ires = (uint)convert_int_rte(scale*lutVal);
	printf("scale:%f,ires:%d\t",scale,ires); 

    //lut[(ty * tiles.x + tx) * srcStep + tid ] =
       //convert_uchar(clamp(ires, (uint)0, (uint)255));
		    lut[(ty * tiles.x + tx) * 256 + tid ] =
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
    dst[mad24(y, srcStep, x )] = convert_uchar(clamp(ires, (uint)0, (uint)255));
}