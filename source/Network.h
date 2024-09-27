#pragma once

#include "SPIRAL_includes.h"
#include "Node.h"



class Network 
{
public:
	
	static float KC;
	static float KN;


	Network(int _datapointSize, int _labelSize, int nLayers = 0, int* sizes = nullptr);

	~Network();


	void learn(float* _datapoint, float* _label, int nSteps);

	void evaluate(float* _datapoint, int nSteps);

	int getNNodes() { return (int)nodes.size(); }

	float computeTotalActivationEnergy();

	// updated only at the end of a evaluate() call
	float* output;

private:

	int datapointSize, labelSize;

	// the first datapointSize nodes correspond to the datapoint, the labelSize next to the label.
	std::vector<Node*> nodes;

	void setActivities(float* _datapoint = nullptr, float* _label = nullptr); 


	bool dynamicTopology;
	std::vector<Node*> nodesOverKC;
	void topologicalOperations();

};