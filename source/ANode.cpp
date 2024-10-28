#include "ANode.h"
#include <algorithm>
#include <limits>


// Recommended values:
float ANode::wReg = 1.0f;

float ANode::wPriorStrength = 1.0f;


float ANode::observationImportance = 1.0f;
float ANode::certaintyDecay = 1.0f;

float ANode::xReg = 1.0f;


ANode::ANode(Assembly* _parentAssembly) : parentAssembly(_parentAssembly)
{

	parents.resize(0); 
	inParentsListIDs.resize(0); 
	children.resize(0);


	
	nPossibleActivations = 1.0f;
	nActivations = nPossibleActivations * parentAssembly->targetFrequency;

	b_mean = NORMAL_01 * .01f + parentAssembly->targetFrequency;
	b_precision = wPriorStrength;
	b_variate = b_mean;

	w_means.resize(0);
	w_precisions.resize(0);
	w_variates.resize(0);

	mu = b_mean;
	x = 0;

	localXReg = xReg;

	isFree = false;
}


void ANode::updateActivation()
{
	// E(1) - E(0)
	float deltaE = 0.f; 


	// to match the target density of active neurons in the assembly.
	float _r = (float)parentAssembly->nActiveNodes + 1.f - x;
	float l1 = logf(1.f + 1.f / _r);
	float _r2 = (_r + 1.f) / ((float) (parentAssembly->nNodes * parentAssembly->nNodes));
	float l2 = logf(_r * _r2);
	deltaE += parentAssembly->densityStrength * l1 * (l2 - 2.f * parentAssembly->targetDensity);


	// to match the target frequency of activation of this neuron across time
	_r = nActivations + 1.f;
	l1 = logf(1.f + 1.f / _r);
	_r2 = (_r + 1.f) / powf((float)nPossibleActivations + 1.0f, 2.f);
	l2 = logf(_r * _r2);
	deltaE += parentAssembly->frequencyStrength * l1 * (l2 - 2.f * parentAssembly->targetFrequency);


	for (int k = 0; k < children.size(); k++)
	{
		ANode& c = *children[k];
		if (c.isFree) [[unlikely]] {continue; } // more efficient to ignore ?

		deltaE += w_variates[k] * ( (1.f - 2.0f * x) * w_variates[k] * (1.0f + c.localXReg) + 2.0f * (c.localXReg - c.x + c.mu));

	}

	float a = 1.0f, b = .00f;
	float newX = deltaE < .0f ? 1.0f : 0.0f; // x=1 is preferred if it reduces the energy, i.e. deltaE = (E1 - E0) < 0. Otherwise x = 0.
	newX = abs(deltaE) > b ? newX : x;

	//float newX = deltaE>.0f ? 1.0f : 0.0f; // TODO proba


	if (x == newX) { return; } // TODO [[likely]] ?

	float delta = newX - x;
	x = newX;
	parentAssembly->nActiveNodes += (int) delta;

	for (int k = 0; k < children.size(); k++)
	{
		children[k]->mu += delta * w_variates[k];
		if (children[k]->isFree) [[unlikely]] {children[k]->x = children[k]->mu; } // more efficient to ignore ?
	}

}



void ANode::setTemporaryWB()
{
	//if ((x == .0f) && !updateWIfXis0) { return; };

	float s1 = 1.0f / b_precision;
	float s2 = b_mean;


	for (int i = 0; i < parents.size(); i++)
	{
		
		float xi = parents[i]->x;
		if (xi == 0.f)  {continue;}  // TODO faster not to take the branch and remove the if ? [[likely]] ?

		int id = inParentsListIDs[i];
		float t1 = xi / (parents[i]->w_precisions[id] + wReg);

		s1 += xi * t1;
		s2 += t1 * parents[i]->w_means[id] * parents[i]->w_precisions[id];
	}


	float epsilon = (x - s2) / (1.f + s1);
	mu = x - epsilon;


	b_variate = epsilon / b_precision + b_mean;


	for (int i = 0; i < parents.size(); i++)
	{
		float xi = parents[i]->x;
		if (xi == 0.f) { continue; }  // TODO [[likely]] ?

		int id = inParentsListIDs[i];
		float tau_i = parents[i]->w_precisions[id];

		parents[i]->w_variates[id] = (epsilon * xi + tau_i * parents[i]->w_means[id]) / (tau_i + wReg);
	}
}



void ANode::calcifyWB()
{
	//if ((x == .0f) && !updateWIfXis0) { return; };

	for (int k = 0; k < children.size(); k++)
	{
		// TODO faster with a branch ?
		w_precisions[k] = w_precisions[k] * (1.0f - certaintyDecay * x) + observationImportance * x;
		w_means[k] = w_variates[k];
	}

	b_precision = b_precision * (1.0f - certaintyDecay) + observationImportance;
	b_mean = b_variate;

	nPossibleActivations++;
	nActivations += x;
}



void ANode::setActivation(float newX)
{
	float delta = newX - x;
	x = newX;
	
	for (int k = 0; k < children.size(); k++)
	{
		children[k]->mu += delta * w_variates[k]; 

		// needed if the datapoint assembly is a parent of the label one (which should not happen)
		if (children[k]->isFree) {children[k]->x = children[k]->mu; }  
	}
}

void ANode::prepareToReceivePredictions()
{
	mu = b_variate;
}

void ANode::transmitPredictions()
{
	for (int k = 0; k < children.size(); k++)
	{
		children[k]->mu += x * w_variates[k];

		if (children[k]->isFree) { children[k]->x = children[k]->mu; }
	}
}


void ANode::addParents(ANode** newParents, int* newInParentIDs, int nNewParents)
{
	parents.insert(parents.end(), newParents, newParents + nNewParents);
	inParentsListIDs.insert(inParentsListIDs.end(), newInParentIDs, newInParentIDs + nNewParents);
}

void ANode::addChildren(ANode** newChildren, int nNewChildren)
{
	children.insert(children.end(), newChildren, newChildren + nNewChildren);

	w_variates.resize(children.size());
	w_means.resize(children.size());
	w_precisions.resize(children.size());

	for (int i = (int) children.size() - nNewChildren; i < children.size(); i++)
	{
		w_variates[i] = .01f * NORMAL_01;
		w_means[i] = w_variates[i];
		w_precisions[i] = wPriorStrength;
	}
}
