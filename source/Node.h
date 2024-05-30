#pragma once

#include "SPIRAL_includes.h"

class Node 
{
public:
	static float priorStrength;
	static float activationDescentStepSize;
	static float observationImportance;
	static float weightRegularization;
	static float certaintyDecay;

	int nParents, nChildren;

	// Allocated size of the array.
	int parentArraySize;

	Node** parents, **children;
	int* inChildID;

	float bx_variate;
	float bx_mean;
	float bx_precision;
	
	float* wx_variates;
	float* wx_means;
	float* wx_precisions;

	float x, fx, tau, epsilon;

	Node(int _nChildren, Node** _children);
	~Node();

	void asynchronousActivationGradientStep();

	void updateIncomingXWBvariates();
	void learnIncomingXWBvariates();

	void computeEpsilon();

	//void synchronousStep();
	//void synchronousStep();
};