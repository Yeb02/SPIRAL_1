#pragma once

#include "SPIRAL_includes.h"

// Be careful as epsilons are never completely recomputed and need be changed everytime any quantity involved in their
// formula is updated: the bias/activation, or the parents weigths/activations. Same for dynamic taus.
class Node 
{
public:

	static float xlr;

	static float wxlr; 
	static float wtlr;

	static float lambda1;
	static float lambda2;

	
	static float xReg; 
	static float wxReg;
	static float wtReg;
	static float btReg;


	static float wxPriorStrength;
	static float wtPriorStrength;

	static float observationImportance;
	static float certaintyDecay;


	int nChildren;
	Node **children;
	
	std::vector<Node*> parents;
	std::vector<int> inParentsListIDs;

	float bx_variate;
	float bx_mean; 
	float bx_precision;
	
	// outgoing weights predicting the children's activations
	float* wx_variates;
	float* wx_means; 
	float* wx_precisions;

#ifdef DYNAMIC_PRECISIONS
	float bt_variate;
	float bt_mean;
	float bt_precision;

	// outgoing weights computing the children's tau
	float* wt_variates;
	float* wt_means;
	float* wt_precisions;

	// the prediction accumulator for tau
	float t;

#endif

	float x, fx, tau, epsilon, mu;


	// for topology
	float accumulatedEnergy; 



	Node(int _nChildren, Node** _children);

	~Node();


	void XGradientStep();

	// deprecated
	void WBGradientStep();


	void setAnalyticalWX();

	void setAnalyticalWT();

	

	// Sets the MAP ('mean') to the variate, and updates the precision
	void calcifyWB();



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