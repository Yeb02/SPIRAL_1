#pragma once

#include "SPIRAL_includes.h"
#include "ANode.h"



class ANetwork 
{
public:
	


	ANetwork(int _datapointSize, int _labelSize, int nLayers = 0, int* sizes = nullptr);

	~ANetwork();


	void learn(float* _datapoint, float* _label, int nSteps);

	void evaluate(float* _datapoint, int nSteps);


	float computeTotalActivationEnergy();

	// updated only at the end of a asynchronousEvaluate() call
	float* output;

private:

	int datapointSize, labelSize;

	bool dynamicTopology;

	// the first datapointSize nodes correspond to the datapoint, the labelSize next to the label.
	std::vector<ANode*> nodes;

	void setActivities(float* _datapoint = nullptr, float* _label = nullptr); 


};