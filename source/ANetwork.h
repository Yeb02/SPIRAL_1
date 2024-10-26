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

private:
	std::vector<int> permutation;

	int datapointSize, labelSize;


	void setActivities(float* _datapoint = nullptr, float* _label = nullptr);

};