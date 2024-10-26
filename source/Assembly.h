#pragma once

#include "SPIRAL_includes.h"

struct Assembly
{
	Assembly(int _nNodes, float _targetDensity, float _targetFrequency, float _densityStrength, float _frequencyStrength) :
		nNodes(_nNodes), targetDensity(_targetDensity), targetFrequency(_targetFrequency), densityStrength(densityStrength), frequencyStrength(frequencyStrength)
	{
		nActiveNodes = 0;
		firstNodeID = -1;
	};

	int nNodes;

	int nActiveNodes;
	int firstNodeID; // The id of the first node of the assembly in the parent ANetwork's nodes array. The nNodes are contiguous.

	float targetDensity;
	float densityStrength;

	float targetFrequency;
	float frequencyStrength;
};