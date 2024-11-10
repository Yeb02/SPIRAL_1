#pragma once

#include "SPIRAL_includes.h"
#include "Node.h"


class Network 
{
public:

	Network(int _datapointSize, int _labelSize);

	~Network();


	void learn(float* _datapoint, float* _label, int nSteps);

	void evaluate(float* _datapoint, int nSteps);

	float computeTotalActivationEnergy();

	// updated only at the end of a evaluate() call
	float* output;


	// Should be private:
	// the first datapointSize nodes correspond to the datapoint, the labelSize next to the label.
	std::vector<Node*> nodes;

	void addGroup(int nNodes);
	void addConnexion(int originGroup, int destinationGroup);

	void initialize();
	void readyForLearning();
	void readyForTesting();

private:

	bool isInitialized; // True after a call to initialize(). Required before switching into learning mode or testing mode
	bool learningMode;	// True if in learning mode, false if in testing mode. False if neither.
	bool testingMode;	// False if in learning mode, true if in testing mode. False if neither.


	std::vector<int> groupSizes;	 // Number of nodes per group
	std::vector<int> groupOffsets;   // Id of the FIRST node of the group in the "nodes" vector.

#ifdef FREE_NODES
	std::vector<int> freeGroups;
#endif

	int datapointSize, labelSize;

	std::vector<int> permutation;

	void setActivities(float* _datapoint = nullptr, float* _label = nullptr); 

};