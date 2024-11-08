#pragma once

#include "SPIRAL_includes.h"

// Be careful as epsilons are never completely recomputed and need be changed everytime any quantity involved in their
// formula is updated: the bias/activation, or the parents weigths/activations. Same for dynamic taus.
class Node 
{
public:

	static float xlr;
	
	static float xReg; 
	static float wxReg;

	static float wxPriorStrength;

	static float observationImportance;
	static float certaintyDecay;

	float localXReg; // set at 0 for observation/label nodes by the parent network, xReg otherwise

	bool isFree; // set to true for nodes that must be inferred and do not have children. Typically the label.

	std::vector<Node*> children;
	std::vector<Node*> parents;
	std::vector<int> inParentsListIDs;

	float bx_variate;
	float bx_mean; 
	float bx_precision;
	
	// outgoing weights predicting the children's activations
	std::vector<float> wx_variates;
	std::vector<float> wx_means;
	std::vector<float> wx_precisions;


	float x, fx, tau, epsilon, mu;


	void compute_sw();

	Node(int _nChildren, Node** _children, int _nCoParents);

	~Node() {};


	void XGradientStep();

	void analyticalXUpdate();


	void setAnalyticalWX();


	// Sets the MAP ('mean') to the variate, and updates the precision. Also updates the energies ahead of topological operations.
	void calcifyWB();


	// For benchmarking purposes
	void predictiveCodingWxGradientStep();

	// Updates the children's predicted quantities (mu and t) as well, which requires them to be up to date 
	// relative to the current parameters ! 
	// And a call to computeLocalQuantities must be performed afterwards by the children !
	void setActivation(float newX);


	// sets up (i.e. intitializes accumulators with the biases) this node to receive and 
	// make sense of prediction information regarding its mu and t
	void prepareToReceivePredictions();

	// sends relevant information to the children for their mu and tau
	void transmitPredictions();

	// Sets all values depending directly on the raw predictions that the parent sent:
	// epsilon, tau.
	void computeLocalQuantities();

};