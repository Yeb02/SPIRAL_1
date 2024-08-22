#pragma once

#include "SPIRAL_includes.h"
#include "Node.h"



class Network 
{
public:
	


	Network(int _datapointSize, int _labelSize, int _nLayers = 0, int* _sizes = nullptr);

	~Network();


	void asynchronousLearn(float* _datapoint, float* _label, int nSteps);

	void asynchronousEvaluate(float* _datapoint, int nSteps);


	void synchronousLearn(float* _datapoint, float* _label, int nSteps);

	void synchronousEvaluate(float* _datapoint, int nSteps);



	float computeTotalActivationEnergy();

	// updated only at the end of a asynchronousEvaluate() call
	float* output;

private:

	int datapointSize, labelSize;

	bool dynamicTopology;

	// used only if fixed topology is enforced
	int nLayers, * sizes;

	// the first datapointSize nodes correspond to the datapoint, the labelSize next to the label.
	std::vector<Node*> nodes;

	void setActivities(float* _datapoint = nullptr, float* _label = nullptr); 


};