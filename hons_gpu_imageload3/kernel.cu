#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdafx.h"
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>

#include "kernel.cuh"
#include "RunTimer.h"

using namespace std;

RunTimer rt;
RunTimer execute;

__device__ bool getPosKernel(int x, int y, bool *d_pixBinaryMap, int d_width)
{
	return d_pixBinaryMap[x + y*d_width];
}

__device__ int findHazardKernel(int x, int y, int x_delta, int y_delta, bool *d_pixBinaryMap, int d_width, int d_height)
{
	int count = 0;
	do
	{
		x += x_delta;
		y += y_delta;
		count++;
	} while (0 <= x && x <= d_width && 0 <= y && y <= d_height && !getPosKernel(x, y, d_pixBinaryMap, d_width));
	return count - 1;
}

__global__ void kernel(bool *d_pixBinaryMap, int *d_pixHazardMap, int d_width, int d_height)
{
	const int compassDirections = 16;
	int directions[compassDirections];
	int min_element;

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < d_width && y < d_height)
	{
		directions[0] = findHazardKernel(x, y, 0, -1, d_pixBinaryMap, d_width, d_height);		// N

		directions[1] = findHazardKernel(x, y, 1, -2, d_pixBinaryMap, d_width, d_height);		// NNE
		directions[2] = findHazardKernel(x, y, 1, -1, d_pixBinaryMap, d_width, d_height);		// NE
		directions[3] = findHazardKernel(x, y, 2, -1, d_pixBinaryMap, d_width, d_height);		// ENE

		directions[4] = findHazardKernel(x, y, 1, 0, d_pixBinaryMap, d_width, d_height);		// E

		directions[5] = findHazardKernel(x, y, 2, 1, d_pixBinaryMap, d_width, d_height);		// ESE
		directions[6] = findHazardKernel(x, y, 1, 1, d_pixBinaryMap, d_width, d_height);		// SE
		directions[7] = findHazardKernel(x, y, 1, 2, d_pixBinaryMap, d_width, d_height);		// SSE

		directions[8] = findHazardKernel(x, y, 0, 1, d_pixBinaryMap, d_width, d_height);		// S

		directions[9] = findHazardKernel(x, y, -1, 2, d_pixBinaryMap, d_width, d_height);		// SSW
		directions[10] = findHazardKernel(x, y, -1, 1, d_pixBinaryMap, d_width, d_height);		// SW
		directions[11] = findHazardKernel(x, y, -2, 1, d_pixBinaryMap, d_width, d_height);		// WSW

		directions[12] = findHazardKernel(x, y, -1, 0, d_pixBinaryMap, d_width, d_height);		// W

		directions[13] = findHazardKernel(x, y, -2, -1, d_pixBinaryMap, d_width, d_height);		// WNW
		directions[14] = findHazardKernel(x, y, -1, -1, d_pixBinaryMap, d_width, d_height);		// NW
		directions[15] = findHazardKernel(x, y, -1, -2, d_pixBinaryMap, d_width, d_height);		// NNW

		min_element = directions[0];
		for (int a = 1; a < compassDirections; a++)
		{
			if (directions[a] < min_element)
			{
				min_element = directions[a];
			}
			else
			{
				continue;
			}
		}

		d_pixHazardMap[x + y*d_width] = min_element * 5;
	}
}

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		cout << "\n # Usage: " << argv[0] << " <filename>\n" << endl;

		cout << " > Press [ENTER] to end the program..." << endl;
		getchar();

		return 1;
	}
	else
	{
		//
		execute.startTimer("EXECUTION TIMER");

		cout << "\n # HONS #GPU COMPONENT \n" << endl;

		//
		//rt.startTimer("Image Input");

		string filename = "images/";
		filename += argv[1];

		if (loadImage(filename))
		{

			//rt.endTimer("Image Input");

			//
			//rt.startTimer("Create Binary and Number of Hazard Maps");

			create_pixBinaryMap();
			pixHazardMap = new int[width*height];

			//rt.endTimer("Create Binary and Number of Hazard Maps");

			// //
			rt.startTimer("CUDA Algorithm");

			cout << " # RUNNING ALGORITHM \n" << endl;
			cudaWrapper();
			cout << "\n ## CUDA Wrapper Complete!\n" << endl;

			rt.endTimer("CUDA Algorithm");

			//
			//rt.startTimer("Image Output");

			cout << " # SAVING HAZARD IMAGE \n" << endl;
			saveHazardImage(filename);

			//rt.endTimer("Image Output");

			execute.endTimer("EXECUTION TIMER");

			cout << "\n > Press [ENTER] to end the program..." << endl;
			getchar();

			return 0;
		}
		else
		{
			cout << "\n > Press [ENTER] to end the program..." << endl;
			getchar();

			return 1;
		}
	}
}

cudaError_t cudaWrapper()
{
	bool	*d_pixBinaryMap;
	int		*d_pixHazardMap;
	int		*d_width, *d_height;
	dim3 dimGrid(128, 128);
	dim3 dimBlock(32, 32);
	cudaError_t cudaStatus;

	//pixHazardMap = new int[width*height];

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, " X cudaSetDevice failed!");
		goto Error;
	}
	else
	{
		printf(" + cudaSetDevice success!\n");
	}

	// // TIMER START
	rt.startTimer("CUDA Memory Allocation");

	cudaStatus = cudaMalloc(&d_width, sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, " X cudaMalloc d_width failed!");
		goto Error;
	}
	else
	{
		printf(" + cudaMalloc d_width success!\n");
	}

	cudaStatus = cudaMalloc(&d_height, sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, " X cudaMalloc d_height failed!");
		goto Error;
	}
	else
	{
		printf(" + cudaMalloc d_height success!\n");
	}

	cudaStatus = cudaMalloc(&d_pixBinaryMap, width*height*sizeof(bool));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, " X cudaMalloc d_pixBinaryMap failed!");
		goto Error;
	}
	else
	{
		printf(" + cudaMalloc d_pixBinaryMap success!\n");
	}

	cudaStatus = cudaMalloc(&d_pixHazardMap, width*height*sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, " X cudaMalloc d_pixHazardMap failed!");
		goto Error;
	}
	else
	{
		printf(" + cudaMalloc d_pixHazardMap success!\n");
	}

	rt.endTimer("CUDA Memory Allocation");

	// // TIMER START
	rt.startTimer("CUDA Memory Copy - Host To Device");

	cudaStatus = cudaMemcpy(d_width, &width, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, " X cudaMemcpy d_width failed!");
		goto Error;
	}
	else
	{
		printf(" + cudaMemcpy d_width success!\n");
	}

	cudaStatus = cudaMemcpy(d_height, &height, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, " X cudaMemcpy d_height failed!");
		goto Error;
	}
	else
	{
		printf(" + cudaMemcpy d_height success!\n");
	}

	cudaStatus = cudaMemcpy(d_pixBinaryMap, pixBinaryMap, width*height*sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, " X cudaMemcpy d_pixBinaryMap failed!");
		goto Error;
	}
	else
	{
		printf(" + cudaMemcpy d_pixBinaryMap success!\n");
	}

	cudaStatus = cudaMemcpy(d_pixHazardMap, pixHazardMap, width*height*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, " X cudaMemcpy d_pixHazardMap failed! \t %s \t %d \n", cudaGetErrorString(cudaStatus), cudaGetErrorName(cudaStatus));
		goto Error;
	}
	else
	{
		printf(" + cudaMemcpy d_pixHazardMap success!\n");
	}

	rt.endTimer("CUDA Memory Copy - Host To Device");

	// //  TIMER START
	rt.startTimer("Kernel Launching");

	kernel << < dimGrid, dimBlock >> >(d_pixBinaryMap, d_pixHazardMap, width, height);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, " X kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	else
	{
		printf(" + kernel launch success!\n");
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, " X cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
		goto Error;
	}
	else
	{
		printf(" + cudaDeviceSynchronize success!\n");
	}

	rt.endTimer("Kernel Launching");

	// // TIMER START
	rt.startTimer("CUDA Memory Copy - Device To Host");

	cudaStatus = cudaMemcpy(pixHazardMap, d_pixHazardMap, width*height*sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, " X cudaMemcpy d_pixHazardMap return failed! \t %s \t %d \n", cudaGetErrorString(cudaStatus), cudaGetErrorName(cudaStatus));
		goto Error;
	}
	else
	{
		printf(" + cudaMemcpy d_pixHazardMap return success!\n");
	}

	rt.endTimer("CUDA Memory Copy - Device To Host");

Error:
	cudaFree(d_pixBinaryMap);
	cudaFree(d_pixHazardMap);
	cudaFree(d_width);
	cudaFree(d_height);

	return cudaStatus;
}

bool loadImage(string filename)
{
	filename += ".ppm";
	string head, wid, ht, colrs;
	string r, g, b;
	ifstream in;

	in.open(filename);

	if (in.is_open())
	{
		cout << " - " << filename << " is open" << endl;

		getline(in, head);
		header = head;
		cout << " - magic header: " << head << '\n';

		getline(in, wid);
		width = atoi(wid.c_str());
		cout << " - width px: " << width;

		getline(in, ht);
		height = atoi(ht.c_str());
		cout << " height px: " << height << '\n';

		getline(in, colrs);
		colourRange = atoi(colrs.c_str());
		cout << " - colour range: " << colrs << '\n';

		pixArray = new pixVals[width*height];

		for (int j = 0; !in.eof() && j < width*height; j++)
		{
			getline(in, r);
			pixArray[j].R = atoi(r.c_str());
			getline(in, g);
			pixArray[j].G = atoi(g.c_str());
			getline(in, b);
			pixArray[j].B = atoi(b.c_str());
		}

		cout << endl;
		in.close();
		return true;
	}
	else
	{
		cerr << " ## ERROR ## Cannot open file: " << filename << "\n\n" << endl;
		return false;
	}
}

void create_pixBinaryMap()
{
	pixBinaryMap = new bool[width*height];

	for (int i = 0; i < width*height; i++)
	{
		if (pixArray[i].R == colourRange)
		{
			pixBinaryMap[i] = true;
		}
		else if (pixArray[i].G == colourRange)
		{
			pixBinaryMap[i] = false;
		}
	}
	cout << " ## Binary Map created!" << endl;
}

void saveHazardImage(string filename)
{
	filename += "_OUT.ppm";
	ofstream fout(filename);
	cout << " ## File output: " << filename << endl;

	fout << "P3\n";
	fout << width << "\n";
	fout << height << "\n";
	fout << "255\n";

	for (unsigned int i = 0; i < width*height; i++)
	{
		if (pixBinaryMap[i])
		{
			fout << "100" << "\n";
			fout << "0" << "\n";
			fout << "0" << "\n";
		}
		else if (pixHazardMap[i] >= 255)
		{
			fout << "255" << "\n";
			fout << "255" << "\n";
			fout << "255" << "\n";
		}
		else
		{
			fout << pixHazardMap[i] << "\n";
			fout << pixHazardMap[i] << "\n";
			fout << pixHazardMap[i] << "\n";
		}
	}

	fout << flush;
	fout.close();

	cout << "\n ## Output file '" << filename << "' saved!" << endl;
}
