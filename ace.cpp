#include "stdafx.h"
#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <fstream>
#include <assert.h>
#include<FreeImage.h>
//#include "CLUtil.hpp"
//#include "SDKBitMap.hpp"
//#include"FreeImage.h"

#ifdef _DEBUG 

#pragma comment(lib,"FreeImage.lib")

#else 

#pragma comment(lib,"FreeImage.lib")

#endif 



#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.1"
#define PROGRAM_FILE "acet.cl"
#define INPUT_IMAGE  "InPutImage.bmp"; 


#define OUTPUT_IMAGE "ace_Output.bmp"
#define GROUP_SIZE 256
#pragma OPENCL EXTENSION cl_amd_printf : enable


#ifndef min
#define min(a, b)            (((a) < (b)) ? (a) : (b))
#endif

#define SUCCESS 0
#define FAILURE 1
using namespace std;
using std::string;



static inline int check_err(int err, const char *s )
{
	if (err < 0) {
		perror(s);
		exit(1);
	}
	return 0; 
}



int main(int argc, char* argv[])
{
	size_t globalSizek1[2];
	size_t localSizek1[2];
	size_t globalSizek2[2];
	size_t localSizek2[2];
	cl_int  tiles[2];
	cl_int tileSize[2];
	cl_float clip;
//	cl_event eventListL[3];
	cl_float lutScale;
	//char* outData;           /**< Output data buffer in host ptr */
	//char* inData; 
	//char* pixelData;       /**< Pointer to image data */
	//cl_uint pixelSize;                  /**< Size of a pixel in BMP format> */



	// 初始化
	FreeImage_Initialise(TRUE);
	// 文件名
	const char* imageFile = "t.jpg";
	const char* saveFile = "t_out.jpg";
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	// 获取图片格式
	/* 此处需要特别注意，即使后缀名是.png并不代表图片格式就真是PNG格式，这里先通过FreeImage_GetFileType函数获取图片格式，
	然后再进行加载，否则，也许会出现加载失败的情况。
	*/
	fif = FreeImage_GetFileType(imageFile);
	if (fif == FIF_UNKNOWN)
		fif = FreeImage_GetFIFFromFilename(imageFile);

	FIBITMAP *bitmap1 = NULL;
	FIBITMAP *bitmap2 = NULL;
	if ((fif != FIF_UNKNOWN) && FreeImage_FIFSupportsReading(fif)){
		bitmap1 = FreeImage_Load(fif, imageFile, PNG_DEFAULT);
	}
	if (!bitmap1){
		fprintf(stderr, "Fail to Load Image!\n");
		exit(-1);
	}
	else{
		FreeImage_Save(fif, bitmap1, saveFile, PNG_DEFAULT);
		bitmap2 = FreeImage_Load(fif, saveFile, PNG_DEFAULT);
		if (!bitmap2){
			fprintf(stderr, "Fail to Load saved Image!\n");
			exit(-1);
		}
	}
	// 获取影像的宽高，都以像素为单位
	int width = FreeImage_GetWidth(bitmap1);
	int height = FreeImage_GetHeight(bitmap1);

	// 获取总共的像素数目
	int pixel_num = width*height;

	// 获取保存每个像素的字节数 这里为3,分别为RGB
	unsigned int byte_per_pixel = FreeImage_GetLine(bitmap1) / width;
	printf("Width:%d\t Height:%d\t 像素总数:%d\t 每像素字节数:%d\n", width, height, pixel_num, byte_per_pixel);
	// 获取保存图片的字节数组
	unsigned char *bits1 = FreeImage_GetBits(bitmap1);
	unsigned char *bits2 = FreeImage_GetBits(bitmap2);



	// get width and height of input image
	localSizek1[0] = 32;
	localSizek1[1] = 8;
	localSizek2[0] = 32;
	localSizek2[1] = 8;
	globalSizek1[0] = 256;
	globalSizek1[1] = 64;
	globalSizek2[0] = width*3;
	globalSizek2[1] = height*3;
	size_t globalSize3[2] = {width,height}; 
	size_t localSize3[2] = {32,8}; 

	tiles[0] = 8;
	tiles[1] = 8;
	tileSize[0] = 1200;
	tileSize[1] = 900;
	lutScale = ((float)255.0) / (float)(tileSize[0] * tileSize[1]);
	printf("scale_host: %f\n", lutScale);
	int bufferSize = width*height*3;
	clip = 0.05f;
	int clipLimit =clip*tileSize[0] * tileSize[1]; 
	printf("clipLimit:%d,tilesize:%d,%d\t", clipLimit,tileSize[0],tileSize[1]);



	//inData = (char*)malloc(width * height * 3);
	//memcpy(inData, pixelData, width * height * 3);
	//outData = (char*)malloc(width * height * sizeof(char)*3);



	/* Host/device data structures */
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_int err;
	/* Program/kernel data structures */
	cl_program program;
	FILE *program_handle;
	char *program_buffer, *program_log;
	size_t program_size, log_size;


	/* Identify a platform */
	err = clGetPlatformIDs(1, &platform, NULL);
	if (err < 0) {
		perror("Couldn't find any platforms");
		exit(1);
	}

	/* Access a device */
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if (err < 0) {
		perror("Couldn't find any devices");
		exit(1);
	}
	/* Create the context */
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if (err < 0) {
		perror("Couldn't create a context");
		exit(1);
	}
	/* Read program file and place content into buffer */
	program_handle = fopen(PROGRAM_FILE, "rb");
	if (program_handle == NULL) {
		perror("Couldn't find the program file");
		exit(1);
	}

	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	rewind(program_handle);
	program_buffer = (char*)malloc(program_size + 1);
	program_buffer[program_size] = '\0';
	fread(program_buffer, sizeof(char), program_size, program_handle);
	fclose(program_handle);


	/* Create program from file */
	program = clCreateProgramWithSource(context, 1,
		(const char**)&program_buffer, &program_size, &err);
	if (err < 0) {
		perror("Couldn't create the program");
		exit(1);
	}
	free(program_buffer);
	program_buffer = NULL; 

	/* Build program */
	err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (err < 0) {

		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
				0, NULL, &log_size);
		program_log = (char*)malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			log_size + 1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		exit(1);
	}


cl_int status = CL_SUCCESS;

#pragma warning( disable : 4996 )
	/* Create a CL command queue for the device*/
	queue = clCreateCommandQueue(context, device, 0, &err);
	if (err < 0) {
		perror("Couldn't create the command queue");
		exit(1);
	}


	/* Create kernel for the mat_vec_mult function */
	cl_kernel kernelRgb2hsv = clCreateKernel(program, "rgb2hsv", &err);
	if (err < 0) {
		perror("Couldn't create the kernel");
		exit(1);
	}

	/* Create kernel for the mat_vec_mult function */
	cl_kernel kernelHsv2rgb = clCreateKernel(program, "hsv2rgb", &err);
	if (err < 0) {
		perror("Couldn't create the kernel");
		exit(1);
	}

	/* Create kernel for the mat_vec_mult function */
	cl_kernel kernelCalLut = clCreateKernel(program, "calLut", &err);
	if (err < 0) {
		perror("Couldn't create the kernel");
		exit(1);
	}

	/* Create kernel for the mat_vec_mult function */
	cl_kernel kernelMapLut = clCreateKernel(program, "mapLut", &err);
	if (err < 0) {
		perror("Couldn't create the kernel");
		exit(1);
	}


	// create meorybuffer 
	cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufferSize, bits1, &err);
	if (err < 0) {
		perror("Couldn't create inputBuffer object\n");
		exit(1);
	}
	cl_mem hsv_in = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize, NULL, &err);
	if (err < 0) {
		perror("Couldn't create hsv object\n");
		exit(1);
	}

	cl_mem hsv_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, NULL, &err);
	if (err < 0) {
		perror("Couldn't create inputBuffer object\n");
		exit(1);
	}

	int histSize = tiles[0] * tiles[1] * 256;
	cl_mem hist = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(char)*histSize, NULL, &err);
	if (err < 0) {
		perror("Couldn't create histBuffer object\n");
		exit(1);
	}
	cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, NULL, NULL);



	// Set appropriate arguments to the kernels mapLut
	status = clSetKernelArg(kernelRgb2hsv, 0, sizeof(cl_mem), &input);
	check_err(status, "mapLut kernel arg failed,input");
	status = clSetKernelArg(kernelRgb2hsv, 1, sizeof(cl_int), &width);
	check_err(status, "mapLut kernel arg failed,width");
	status = clSetKernelArg(kernelRgb2hsv, 2, sizeof(cl_mem), &hsv_in);
	check_err(status, "mapLut kernel arg failed,output");
	status = clSetKernelArg(kernelRgb2hsv, 3, sizeof(cl_mem), &width);
	check_err(status, "mapLut kernel arg failed,hist");
	status = clSetKernelArg(kernelRgb2hsv, 4, sizeof(cl_int), &height);
	check_err(status, "mapLut kernel arg failed,width");

	//cl_int status = CL_SUCCESS; 

	status = clSetKernelArg(kernelCalLut, 0, sizeof(cl_mem), &hsv_in);
	check_err(status, "calLut kernel arg failed,input");
	status = clSetKernelArg(kernelCalLut, 1, sizeof(cl_int), &width);
	check_err(status, "calLut kernel arg failed,width");
	status = clSetKernelArg(kernelCalLut, 2, sizeof(cl_mem), &hist);
	check_err(status, "calLut kernel arg failed,hist");
	status = clSetKernelArg(kernelCalLut, 3, 2 * sizeof(cl_int), &tileSize);
	check_err(status, "calLut kernel arg failed,tilesize");
	status = clSetKernelArg(kernelCalLut, 4, 2 * sizeof(cl_int), &tiles);
	check_err(status, "calLut kernel arg failed,tiles");
	status = clSetKernelArg(kernelCalLut, 5, sizeof(cl_float), &clipLimit);
	check_err(status, "calLut kernel arg failed,clip");


	// Set appropriate arguments to the kernels mapLut
	status = clSetKernelArg(kernelMapLut, 0, sizeof(cl_mem), &hsv_in);
	check_err(status, "mapLut kernel arg failed,input");
	status = clSetKernelArg(kernelMapLut, 1, sizeof(cl_int), &width);
	check_err(status, "mapLut kernel arg failed,width");
	status = clSetKernelArg(kernelMapLut, 2, sizeof(cl_mem), &hsv_out);
	check_err(status, "mapLut kernel arg failed,output");
	status = clSetKernelArg(kernelMapLut, 3, sizeof(cl_mem), &hist);
	check_err(status, "mapLut kernel arg failed,hist");
	status = clSetKernelArg(kernelMapLut, 4, sizeof(cl_int), &width);
	check_err(status, "mapLut kernel arg failed,width");
	status = clSetKernelArg(kernelMapLut, 5, sizeof(cl_int), &height);
	check_err(status, "mapLut kernel arg failed,height");
	status = clSetKernelArg(kernelMapLut, 6, 2 * sizeof(cl_int), &tileSize);
	check_err(status, "mapLut kernel arg failed,tileSize");
	status = clSetKernelArg(kernelMapLut, 7, 2 * sizeof(cl_int), &tiles);
	check_err(status, "mapLut kernel arg failed,tiles");

	int w = width;
	int h = height;

	// Set appropriate arguments to the kernels mapLut
	status = clSetKernelArg(kernelHsv2rgb, 0, sizeof(cl_mem), &hsv_out);
	check_err(status, "mapLut kernel arg failed,input");
	status = clSetKernelArg(kernelHsv2rgb, 1, sizeof(cl_int), &width);
	check_err(status, "mapLut kernel arg failed,width");
	status = clSetKernelArg(kernelHsv2rgb, 2, sizeof(cl_mem), &output);
	check_err(status, "mapLut kernel arg failed,output");
	status = clSetKernelArg(kernelHsv2rgb, 3, sizeof(cl_mem), &width);
	check_err(status, "mapLut kernel arg failed,hist");
	status = clSetKernelArg(kernelHsv2rgb, 4, sizeof(cl_int), &height);
	check_err(status, "mapLut kernel arg failed,width");




	cl_event event[4]; 

	status = clEnqueueNDRangeKernel(queue, kernelRgb2hsv, 2, NULL, globalSize3, localSize3, 0, NULL,&event[0]);
	clEnqueueWaitForEvents(queue, 1, &event[0]);
	check_err(status, "rgb2hsv kernelNDrange wrong\n");
	// waiti for convert rgb2hsv completed...
	//clEnqueueWaitForEvents(queue,1,&e_r2s); 

	//cl_event e_ace[2]; 
	status = clEnqueueNDRangeKernel(queue, kernelCalLut, 2, NULL, globalSizek1, localSizek1, 0, NULL, &event[1]);
	clEnqueueWaitForEvents(queue, 1, &event[1]);
	check_err(status, "calLut set wrong");

	status = clEnqueueNDRangeKernel(queue, kernelMapLut, 2, NULL, globalSizek2, localSizek2, 0, NULL, &event[2]);
	clEnqueueWaitForEvents(queue, 1, &event[2]);
	check_err(status,"k1 set wrong");

	status = clEnqueueNDRangeKernel(queue, kernelHsv2rgb, 2, NULL, globalSize3, localSize3, 0, NULL, &event[3]);
	//clFlush(queue);
	clFinish(queue); 
	//clEnqueueWaitForEvents(queue,1,&event[3]);
	check_err(status, "k3 set wrong");



	void* out; 
	int outBufferSize = width*height*3;

	//out = (char*)malloc(outBufferSize);
	out = clEnqueueMapBuffer(queue, output, CL_TRUE, CL_MAP_READ, 0, outBufferSize, 0, NULL, NULL, &status);
	check_err(status, "mapBuffer wrong");
	//memcpy(outData, out, bufferSize);
	memcpy(bits2,out,outBufferSize);
	status = clEnqueueUnmapMemObject(queue, output, (void*)out, 0, 0, NULL);

	// Wait for the read buffer to finish execution
	status = clFinish(queue);

	FreeImage_Save(fif, bitmap2, saveFile, PNG_DEFAULT);



	status = clReleaseKernel(kernelCalLut);				//Release kernel.
	status = clReleaseKernel(kernelMapLut);				//Release kerne
	status = clReleaseKernel(kernelRgb2hsv);				//Release kernel.
	status = clReleaseKernel(kernelHsv2rgb);				//Release kerne
	status = clReleaseMemObject(input);		//Release mem object.
	status = clReleaseMemObject(output);
	status = clReleaseMemObject(hsv_in);
	status = clReleaseMemObject(hist);
	status = clReleaseMemObject(hsv_out);


	FreeImage_Unload(bitmap1);
	FreeImage_Unload(bitmap2);
	// 撤销初始化
	FreeImage_DeInitialise();


	/*Step 12: Clean the resources.*/
	status = clReleaseProgram(program);				//Release the program object.
	status = clReleaseCommandQueue(queue);	//Release  Command queue.
	status = clReleaseContext(context);				//Release context.
	std::cout << "Passed!\n";
	getchar(); 
	return SUCCESS;
}
