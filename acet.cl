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


__kernel void calLut(__global __const uchar * src, const int srcStep,__global uchar * lut,const int2 tileSize, const int2 tiles,const int clip1,const int clip2,const int framesize)
{
    __local int smem[512];
	__local int vmem[512];
    int tx = get_group_id(0);
    int ty = get_group_id(1);
    int tid = get_local_id(1) * get_local_size(0)+ get_local_id(0);

    smem[tid]=0;
	vmem[tid]=0; 
    barrier(CLK_LOCAL_MEM_FENCE);
	
	//if(tx==0 && ty==0)
	//	printf("Height:%d",tileSize.y*tiles.y); 

    for (int i = get_local_id(1); i < tileSize.y; i += get_local_size(1))
    {
        __global const uchar* srcPtr =src+mad24(ty * tileSize.y + i, srcStep, tx * tileSize.x );
		__global const uchar* srcPtrv=srcPtr+framesize; 
        for (int j = get_local_id(0); j < tileSize.x; j += get_local_size(0))
        {
            const int data = srcPtr[j];
            atomic_inc(&smem[data]);
			const int vdata=srcPtrv[j]; 
			atomic_inc(&vmem[vdata]); 
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
	
    int tHistVal =smem[tid];
	int vHistVal=vmem[tid];
    barrier(CLK_LOCAL_MEM_FENCE);

	// above test is ok!!


    int clipped = 0;
    if (tHistVal > clip1)
    {
        clipped = tHistVal - clip1;
        tHistVal = clip1;
    }

	int clipped2 = 0;
    if (vHistVal > clip2)
    {
        clipped2 = vHistVal - clip2;
        vHistVal = clip2;
    }
	
	// above is ok !!
	clipped = reduce(smem, clipped, tid);
	clipped2 =reduce(vmem, clipped2, tid);

	//if(tx<=2)
	//printf("calLut,clip:%u,clip-clip2:%u\t",clipped,clipped-clipped2);
	
    int redistBatch = clipped / 256;
	int redistBatch2 = clipped2 / 256;
	
	//if(tx<=2 && ty==0)
	//		printf("redis1:%d,redis2:%d\t",redistBatch,redistBatch2); 
	//above is ok!!

    tHistVal += redistBatch;
	vHistVal += redistBatch2;
	
    int residual = clipped - redistBatch * 256;
	int residual2 = clipped2 - redistBatch2 * 256;
	
	
	if(tid<residual)	
		++tHistVal;
	if(tid<residual2)	
		++vHistVal;
	
    const int lutVal = calc_lut(smem, tHistVal, tid);
	const int lutVal2 = calc_lut(vmem, vHistVal, tid);
	
	//if(tx<=2 && ty==0)
	//	printf("lutv1:%d,lutv2:%d\t",lutVal,lutVal2); 
	//above is ok!!

	int area=tileSize.x*tileSize.y; 
	const float scale=convert_float(255.0/area);	
    uint ires = (uint)convert_int_rte(scale*lutVal);
	uint ires2 = (uint)convert_int_rte(scale*lutVal2);

	//if(tx<=2 && ty==0)
	//	printf("ires:%d,ires2:%d\t",ires,ires2); 
	//above is ok!!

	int lutidx=(ty*tiles.x+tx)*256; 
	int lutLen=tiles.x*tiles.y*256; 
	lut[lutidx + tid] =
        convert_uchar(clamp(ires, (uint)0, (uint)255));
	
	lut[lutLen+lutidx + tid] =
        convert_uchar(clamp(ires2, (uint)0, (uint)255));
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

	const int frameSize=cols*rows; 
	const int idx=mad24(y, srcStep, x ); 
    const int srcVal = src[idx];
	const int srcVal2=src[frameSize+idx];
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	//if(x<=2)
		//printf("srcVal%u,srcVal2:%u\t",srcVal,srcVal2);
	//above is ok !!

    float res = 0;
    res += lut[mad24(ty1 * tiles.x+ tx1, lutStep, srcVal)] * ((1.0f - xa) * (1.0f - ya));
    res += lut[mad24(ty1 * tiles.x + tx2, lutStep, srcVal)] * ((xa) * (1.0f - ya));
    res += lut[mad24(ty2 * tiles.x + tx1, lutStep, srcVal)] * ((1.0f - xa) * (ya));
    res += lut[mad24(ty2 * tiles.x + tx2, lutStep, srcVal)] * ((xa) * (ya));
	uint ires = (uint)convert_int_rte(res);
	
	barrier(CLK_GLOBAL_MEM_FENCE); 
	
	float res2 = 0;
	const int lutLen=tiles.x*tiles.y*256;
    res2 += lut[mad24(ty1 * tiles.x+ tx1, lutStep, srcVal2)+lutLen] * ((1.0f - xa) * (1.0f - ya));
    res2 += lut[mad24(ty1 * tiles.x + tx2, lutStep, srcVal2)+lutLen] * ((xa) * (1.0f - ya));
    res2 += lut[mad24(ty2 * tiles.x + tx1, lutStep, srcVal2)+lutLen] * ((1.0f - xa) * (ya));
    res2 += lut[mad24(ty2 * tiles.x + tx2, lutStep, srcVal2)+lutLen] * ((xa) * (ya));
	uint ires2 = (uint)convert_int_rte(res2);

	barrier(CLK_GLOBAL_MEM_FENCE); 
	
	//if(x<=10)
	//	printf("ires:%u,ires2:%u\t",ires,ires2);
	//above is ok !!
		
    
	int s_idx=mad24(y, srcStep, x);
	int v_idx=mad24(cols,rows,s_idx); 
	int h_idx=mad24(cols,rows,v_idx); 
    dst[s_idx] = convert_uchar(clamp(ires, (uint)0, (uint)255));
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	//dst[s_idx]=src[s_idx];
	dst[v_idx]=convert_uchar_sat(clamp(ires2, (uint)0, (uint)255));
	//dst[v_idx]=src[v_idx];
	dst[h_idx]=src[h_idx];
	barrier(CLK_GLOBAL_MEM_FENCE);

	//if(x<=3)
		//printf("ires:%u,ires2:%u\t",dst[s_idx],dst[v_idx]);
	//above is ok !!
	
	
}

__kernel void rgb2hsv(__global __const uchar * src, const int srcStep, __global uchar * dst,const int cols, const int rows)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
   
    if (x >= cols || y >= rows)
		return; 
	int idx=mad24(y,srcStep*3,x*3); 
	int r = src[idx];
	int g=src[idx+1];
    int b=src[idx+2];
	const float scale=255.f/360.f; 
	

	float h,s,v; 
	int maxv=max(r,max(g,b)); 
	int minv=min(r,min(g,b)); 
	int delt=maxv-minv; 
	float fDelt=60.0f/convert_float(delt); 

	
	if(maxv)
		s=convert_float(delt)/convert_float(maxv); 
	else 
		s=0; 
	v=maxv; 
	if (maxv==minv)
		h=0; 
	else if(maxv==r)
		h=(g-b)*fDelt;
	else if(maxv==g)
		h=(b-r)*fDelt+120.f;
	else if(maxv==b)
		h=(r-g)*fDelt+240.0f; 
	if(h<0) 
		h+=360.f; 


	
	uint h1 = (uint)convert_int_rte(h*scale);
	uint s1 = (uint)convert_int_rte(s*255);
	uint v1 = (uint)convert_int_rte(v);
	
	uint frameSize=mad24(cols,rows,0); 
	uint s_idx=mad24(y,srcStep,x); 
	uint v_idx=frameSize+s_idx; 
	uint h_idx=frameSize+v_idx; 

	dst[h_idx]=convert_uchar_sat(h1); 
	dst[s_idx]=convert_uchar_sat(s1);  
	dst[v_idx]=convert_uchar_sat(v1);  

}

__kernel void hsv2rgb(__global uchar* src, const int srcStep,__global uchar* dst,const int cols, const int rows )
{
	const int tidx=get_global_id(0);
	const int tidy=get_global_id(1); 
	
	if(tidx>=cols || tidy>=rows)
		return; 
	
	int frameSize=mad24(rows,cols,0); 
	int s_idx=mad24(tidy,srcStep,tidx); 
	int v_idx=frameSize+s_idx; 
	int h_idx=frameSize+v_idx; 
	
	float r,g,b; 
    float h =convert_float(src[h_idx]);
	float s =convert_float(src[s_idx])/255;
	float v=convert_float(src[v_idx])/255;
	const float scale=1.412f; 
	
	if(s==0)
	{
		r=g=g=v; 
		return; 
	}
	
	float f=h*scale/60; 
	float hi=floor(f);
	f=f-hi; 
	
	float p=v*(1-s); 
	float q=v*(1-s*f);
	float t=v*(1-s*(1-f));
	if(hi==0.0f||hi==6.0f)
	{
		r=v; 
		g=t; 
		b=p; 
	}else if(hi==1.0f)
	{
		r=q;
		g=v; 
		b=p; 
	}else if(hi==2.0f)
	{
		r=p; 
		g=v; 
		b=t; 
	}else if(hi==3.0f)
	{
		r=p; 
		g=q; 
		b=v;
	}else if(hi==4.0f)
	{
		r=t; 
		g=p;
		b=q; 
	}else
	{
		r=v;
		g=p;
		b=q;
	}

	uint r1 = (uint)convert_int_rte(r*255);
	uint g1 = (uint)convert_int_rte(g*255);
	uint b1 = (uint)convert_int_rte(b*255);

	int idx=mad24(tidy,srcStep*3,tidx*3); 
	dst[idx]=convert_uchar_sat(clamp(r1, (uint)0, (uint)255)); 
	dst[idx+1]=convert_uchar_sat(clamp(g1, (uint)0, (uint)255)); 
	dst[idx+2]=convert_uchar_sat(clamp(b1, (uint)0, (uint)255)); 	 
}
	