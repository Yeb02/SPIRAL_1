#pragma once

#include "SPIRAL_includes.h"

// Be careful as epsilons are never completely recomputed and need be changed everytime any quantity involved in their
// formula is updated: the bias/activation, or the parents weigths/activations. Same for dynamic taus.
class Node 
{
public:
	static float priorStrength;
	static float activationDescentStepSize;
	static float wxDescentStepSize;
	static float observationImportance;
	static float weightRegularization;
	static float certaintyDecay;
	static float potentialConservation;

	int nParents, nChildren;

	// Allocated size of the array.
	int parentArraySize;

	Node** parents, **children;
	int* inChildID;

	float bx_variate;
	float bx_mean; // TODO unify mean and variate
	float bx_precision;
	
	// incoming weights
	float* wx_variates;
	float* wx_means; // TODO unify mean and variate
	float* wx_precisions;

	float x, fx, tau, epsilon;
	float potential, accumulatedEnergy;

	Node(int _nChildren, Node** _children);
	~Node();

	void asynchronousActivationGradientStep();
	void asynchronousWeightGradientStep();

	// sets incoming weigths variates and the bias variate to the analytical optimum given the fixed activations of this node and its parents.
	// Requires lambert's Ws for dynamic taus, so should not be used. "Deprecated"
	void updateIncomingXWBvariates();

	// Sets the MAP ('mean') to the variate. Should soon not be needed anymore  as those will be the same quantity.
	void learnIncomingXWBvariates();

	// Deprecated
	void computeEpsilon();

	void setActivation(float newX);
};