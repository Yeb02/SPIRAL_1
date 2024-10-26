#pragma once

#include "SPIRAL_includes.h"
#include "Assembly.h"


class ANode
{
public:

	static float wReg;

	static float wPriorStrength;

	static float observationImportance;
	static float certaintyDecay;

	static float xReg; //?
	float localXReg; // set at 0 for observation nodes by the parent network, xReg otherwise


	Assembly* parentAssembly;

	std::vector<ANode*> children;
	std::vector<ANode*> parents;

	std::vector<int> inParentsListIDs;

	float b_variate;
	float b_mean;
	float b_precision;

	// outgoing weights predicting the children's activations
	std::vector<float> w_variates;
	std::vector<float> w_means;
	std::vector<float> w_precisions;

	float nActivations, nPossibleActivations;

	float x, mu;

	// true of label nodes at inference time (and of observation nodes at inference after some time, if the task is reconstruction)
	bool isFree;

	ANode(Assembly* _parentAssembly);

	~ANode() {};


	void updateActivation();

	void setTemporaryWB();

	// Sets the MAP ('mean') to the variate, and updates the precision. Also updates the energies ahead of topological operations.
	void calcifyWB();



	//void pruneUnusedConnexions();

	
	// Updates the children's mu as well, which requires them to be up to date 
	// relative to the current parameters ! 
	void setActivation(float newX);


	// sets up (i.e. sets mu = b) this node to receive and 
	// make sense of prediction information regarding its mu
	void prepareToReceivePredictions();

	// sends relevant information to the children for their mu 
	void transmitPredictions();


	void addParents(ANode** newParents, int* newInParentIDs, int nNewParents);

	void addChildren(ANode** newChildren, int nNewChildren);
};