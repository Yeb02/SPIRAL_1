#pragma once

#include "SPIRAL_includes.h"
#include "Node.h"

#include <vector>


class Network 
{
public:
	


	Network(int _datapointSize, int _labelSize);
	~Network();

	void asynchronousLearn(float* _datapoint, float* _label, int nSteps);
	void asynchronousEvaluate(float* _datapoint, int nSteps);

	float computeTotalActivationEnergy();

	// updated only at the end of a asynchronousEvaluate() call
	float* output;

private:
	int datapointSize, labelSize;
	std::vector<Node*> nodes;

	void setDatapoint(float* _datapoint);
	void setLabel(float* _label);

	void initializeEpsilons();

};