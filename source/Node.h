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


	// I tried to take a non-biasing approach to regularization. Hopefully this implementation achieves it, but i am unsure.
	static float xReg;
	static float wxReg;
	static float wtReg;

	static float priorStrength;
	static float observationImportance;
	static float certaintyDecay;


	int nChildren;
	Node **children;

	float bx_variate;
	float bx_mean; 
	float bx_precision;
	
	// outgoing weights predicting the children's activations
	float* wx_variates;
	float* wx_means; 
	float* wx_precisions;

	float x, fx, tau, epsilon, mu;

	// A util for the Network. True if epsilon, mu, tau, ... have the correct value corresponding to the parent's activations and weights.
	bool quantitiesUpToDate;

	Node(int _nChildren, Node** _children);

	~Node();


	// differs from synchronousGradientStep in that it notifies its children of their new predicted activation / tau
	// The order of update is somewhat arbitrarily x then epsilon then b,w 
	void asynchronousGradientStep();

	// A chunk of asynchronousGradientStep that does not change weights, to be used at evaluation time
	// by non datapoint neurons instead of asynchronousGradientStep.
	void asynchronousGradientStep_X_only();

	// A chunk of asynchronousGradientStep that does not change activations, to be used at training time
	// by datapoint and label neurons.
	void asynchronousGradientStep_WB_only();



	// differs from asynchronousGradientStep in that it does not notify its children of their new predicted activation / tau
	// The order of update is somewhat arbitrarily x then epsilon then b,w
	void synchronousGradientStep();


	// Sets the MAP ('mean') to the variate, and updates the precision
	void calcifyWB();


	// Updates the children's predicted quantities as well, which requires them to be up to date relative to the current parameters ! (mu and mu_tau)
	// And a call to computeLocalQuantities must be performed afterwards by the children !
	void setActivation(float newX);



	// These 3 functions must be called in this order, looping over each neuron in the network before calling the next one.
	// They need be called only when there is an external intervention on the network's activations. (network creation counts as such)
	

	// sets up (i.e. zeroes accumulators) this node to receive and make sense of prediction information regarding its mu and tau
	void prepareToReceivePredictions();

	// sends relevant information to the children for their mu and tau
	void transmitPredictions();

	// Transform the predictions from the parents into the effective values: epsilon set to  x - mu, tau computed.
	void computeLocalQuantities(); // set quantities up to date tot true


};