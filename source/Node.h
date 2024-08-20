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


	Node(int _nChildren, Node** _children);
	~Node();

	// differs from synchronousGradientStep in that it notifies its children of their new predicted activation / tau
	// The order of update is somewhat arbitrarily x then epsilon then b,w 
	void asynchronousGradientStep();

	// differs from asynchronousGradientStep in that it does not notify its children of their new predicted activation / tau
	// The order of update is somewhat arbitrarily x then epsilon then b,w
	void synchronousGradientStep();

	// Sets the MAP ('mean') to the variate, and updates the precision
	void calcifyWB();



	// sets up (i.e. zeroes accumulators) this node to receive and make sense of prediction information regarding its mu and tau
	void prepareToReceivePredictions();

	// sends relevant information to the children for their mu and tau
	void transmitPredictions();

	// simply sets epsilon to  x - mu
	void computeEpsilon();

	// does not update the children's quantities ! (mu and tau)
	void setActivation(float newX);
};