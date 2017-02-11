#pragma once

#include <string>

struct pixVals
{
	unsigned int R, G, B;
};

std::string header;
int width, height, colourRange;

pixVals *pixArray;

bool* pixBinaryMap;
int* pixHazardMap;

cudaError_t cudaWrapper();

bool loadImage(std::string);
void create_pixBinaryMap();
void saveHazardImage(std::string);