#pragma once

#include "SPIRAL_includes.h"
#include "ANode.h"
#include "Assembly.h"



class ANetwork
{
public:

	ANetwork(int _datapointSize, int _labelSize);

	~ANetwork();

	void readyForLearning();

	void readyForTesting();

	void addAssembly(Assembly* assembly);


	// p is the connexion probability. The IDs are those in std::vector<Assembly*> assemblies. 
	// p should not be too close to 1 (i.e. < 0.9)
	// Nodes can not connect to themselves, but later it should be enforced in a smarter way than simply 
	// not creating the connexion, for better coalescing. (probably clamp the weight at 0 at all time)
	void addConnexion(int originID, int destinationID, float p);

	void learn(float* _datapoint, float* _label, int nSteps);

	void evaluate(float* _datapoint, int nSteps);

	float computeTotalActivationEnergy();

	// updated only at the end of a evaluate() call
	float* output;


	// Should be private:
	
	// the first datapointSize nodes correspond to the datapoint, the labelSize next to the label.
	std::vector<ANode*> nodes;

	// the first assembly corresponds to the datapoint, the second to the label.
	std::vector<Assembly*> assemblies;


	int getNNodes() { return 0; } // just for interchangability of ANetwork and Network in main.cpp

private:
	std::vector<int> permutation;

	int datapointSize, labelSize;


	void setActivities(float* _datapoint = nullptr, float* _label = nullptr);

};