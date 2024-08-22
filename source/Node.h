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

	static float wxPriorStrength;
	static float wtPriorStrength;
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

	// stored because needs be computed for each incoming weight.
	// = .5 * eps * eps * tau 
	float e;

#endif

	float x, fx, tau, epsilon, mu;

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

	// A chunk of synchronousGradientStep that does not change weights, to be used at evaluation time
	// by non datapoint neurons instead of asynchronousGradientStep.
	void synchronousGradientStep_X_only();

	// A chunk of synchronousGradientStep that does not change activations, to be used at training time
	// by datapoint and label neurons.
	void synchronousGradientStep_WB_only();



	// Sets the MAP ('mean') to the variate, and updates the precision
	void calcifyWB();


	// Updates the children's predicted quantities (mu and t) as well, which requires them to be up to date relative to the current parameters ! 
	// And a call to computeLocalQuantities must be performed afterwards by the children !
	void setActivation(float newX);



	// These 3 functions must be called in this order, looping over each neuron in the network before calling the next one.
	// They need be called only when there is an external intervention on the network's activations. (network creation counts as such)
	

	// sets up (i.e. intitializes accumulators with the biases) this node to receive and make sense of prediction information regarding its mu and t
	void prepareToReceivePredictions();

	// sends relevant information to the children for their mu and tau
	void transmitPredictions();

	// Sets all values depending directly or not on the raw predictions that the parent sent:
	// epsilon, tau, e.
	void computeLocalQuantities();


};