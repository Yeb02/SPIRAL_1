#pragma once

#include "SPIRAL_includes.h"

// Be careful as epsilons are never completely recomputed and need be changed everytime any quantity involved in their
// formula is updated: the bias/activation, or the parents weigths/activations. 
class ANode 
{
public:

	//parameters:

	static float xlr;

	static float xReg;
	static float wReg;

	static float wPriorStrength;

	static float observationImportance;
	static float certaintyDecay;




	int nChildren, nParents;
	ANode **children;
	// using a vector instead of a raw pointer because this array will be lengthened as parents are added one by one during
	// lifetime if DYNAMIC_TOPOLOGY is enabled. Same for inParentsListIDs below.
	std::vector<ANode*> parents; 

	float b_variate;
	float b_mean; 
	float b_precision;
	
	// those are the OUTGOING weights, predicting the children's activations
	float* w_variates;
	float* w_means; 
	float* w_precisions;

	// for each of the parent nodes, this node's pointer's ID in their respective "ANode **children".
	std::vector<int> inParentsListIDs;


	// since epsilon = x-mu, keeping the 3 is a bit redundant, but kept for clarity's sake:

	float x, fx, mu, epsilon;



	ANode(int _nChildren, ANode** _children);

	~ANode();


	void addParent(ANode* parent, int inParentsListID);
	
	// A util for efficient fully connected network creation. Do not call on a node that already has parents.
	void registerInitialParents(ANode** parent, int* inParentsListID, int nParents);



	void updateActivation();

	void updateIncomingWeights();

	void calcifyIncomingWeights();



	// Updates the children's predicted quantities (mu and t) as well, which requires them to be up to date relative to the current parameters ! 
	// And a call to computeLocalQuantities must be performed afterwards by the children !
	void setActivation(float newX);



	// These 3 functions are only called once, at network creation:

	// intitializes the predicted activation mu with the bias
	void prepareToReceivePredictions();

	// sends relevant information to the children for their mu and tau
	void transmitPredictions();

	// Sets all values depending directly or not on the raw predictions that the parent sent:
	// (only epsilon as of yet)
	void computeLocalQuantities();


};