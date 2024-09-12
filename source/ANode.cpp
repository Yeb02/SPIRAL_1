#include "ANode.h"

// Recommended values:

float ANode::xlr = .7f;

float ANode::xReg = .05f;
float ANode::wReg = .05f;

float ANode::wPriorStrength = 1.0f;

float ANode::observationImportance = 1.0f;
float ANode::certaintyDecay = 1.0f;



ANode::ANode(int _nChildren, ANode** _children) :
	nChildren(_nChildren)
{
	

	children = new ANode*[nChildren];
	std::copy(_children, _children + nChildren, children);

	nParents = 0;
	parents.resize(0); // for completeness
	inParentsListIDs.resize(0); // for completeness

	b_mean = NORMAL_01 * .05f;
	b_precision = wPriorStrength;
	b_variate = b_mean;

	w_means = new float[nChildren];
	w_precisions = new float[nChildren];
	w_variates = new float[nChildren];

	std::fill(w_precisions, w_precisions + nChildren, wPriorStrength);
	for (int i = 0; i < nChildren; i++) w_means[i] = NORMAL_01 * .05f;
	std::copy(w_means, w_means + nChildren, w_variates);

	x = NORMAL_01 * .05f;
	fx = std::clamp(x, .0f, 1.f);

	transmitPredictions();
}

ANode::~ANode()
{
	delete[] children;
	delete[] w_variates;
	delete[] w_means;
	delete[] w_precisions;
}




void ANode::addParent(ANode* parent, int inParentsListID) 
{
	parents.push_back(parent);
	inParentsListIDs.push_back(inParentsListID);
}

void ANode::registerInitialParents(ANode** _parents, int* _inParentsListIDs, int _nParents)
{
	nParents = _nParents;
	parents.resize(nParents);
	inParentsListIDs.resize(nParents);

	std::copy(_inParentsListIDs, _inParentsListIDs + nParents, inParentsListIDs.data());
	std::copy(_parents, _parents + nParents, parents.data());
}



void ANode::updateActivation() 
{
	float swv = .0f, sw2 = .0f;
	for (int i = 0; i < nChildren; i++) 
	{
		swv += w_variates[i] * (children[i]->epsilon + w_variates[i] * fx);
		sw2 += w_variates[i] * w_variates[i];
	}

	float xstar = (mu + swv) / (1.f + sw2);
	float clmpxstar = std::clamp(xstar, .0f, 1.f);

	if (xstar != clmpxstar)  // x not in [0, 1], so saturated. As branchless as possible.
	{
		xstar = (((mu-1.f) * clmpxstar - mu * (1.f- clmpxstar)) > 0.f) ? 
			(clmpxstar * xReg + mu)/(1.f + xReg) : 
			clmpxstar;
	}
	x = x * (1.f - xlr) + xstar * xlr;

	float newfx = std::clamp(x, .0f, 1.f);
	if (newfx != fx) // TODO monitor performance, branch misprediction may be more costly than the few operations we avoid
	{
		float deltafx = newfx - fx;
		fx = newfx;
		for (int i = 0; i < nChildren; i++)
		{
			children[i]->mu += w_variates[i] * deltafx;
			children[i]->epsilon = children[i]->x - children[i]->mu;
		}
	}
}


void ANode::updateIncomingWeights() 
{

	float s1 = .0f, s2 = .0f;

	// b:
	s1 = 1.0f / b_precision;

	// first and second formulas seem to work better.
	//float lambdab = wReg * epsilon * epsilon; 
	float lambdab = wReg * epsilon * epsilon * 1.0f * 1.0f; //  (1.0f = parent->fx)
	//float lambdab = wReg; 

	s2 = b_mean + s1 * lambdab;

	// ws:
	for (int i = 0; i < nParents; i++) 
	{
		float fi = parents[i]->fx;
		float fi_div_twi = fi / parents[i]->w_precisions[inParentsListIDs[i]];

		s1 += fi * fi_div_twi;

		float lambdai = wReg * epsilon * epsilon * fi * fi;

		s2 += fi * parents[i]->w_means[inParentsListIDs[i]] + lambdai * fi_div_twi;
	}

	
	epsilon = (x - s2) / (1.f + s1);
	mu = x - epsilon;

	// b:
	b_variate = b_mean + (epsilon + lambdab) / b_precision;

	// ws:
	for (int i = 0; i < nParents; i++)
	{
		float fi = parents[i]->fx;

		// to be rigorous, should use the value of epsilon before the update, but same results so ...
		float lambdai = wReg * epsilon * epsilon * fi * fi;

		parents[i]->w_variates[inParentsListIDs[i]] = parents[i]->w_means[inParentsListIDs[i]] +
			(fi * epsilon + lambdai) / parents[i]->w_precisions[inParentsListIDs[i]];
	}
}


void ANode::calcifyIncomingWeights() 
{
	for (int k = 0; k < nParents; k++)
	{ 
		float epsw = parents[k]->w_variates[inParentsListIDs[k]] - parents[k]->w_means[inParentsListIDs[k]];
		float precw = parents[k]->w_precisions[inParentsListIDs[k]];
		float fxk2 = parents[k]->fx * parents[k]->fx; // measured positive impact.

		float decay = std::max(expf( - certaintyDecay * powf(epsw, 2.0f) * precw), .8f);
		//float decay = std::max(expf( - certaintyDecay * abs(epsw) * sqrtf(precw)), .8f);
		
		parents[k]->w_precisions[inParentsListIDs[k]] = precw * decay + observationImportance * fxk2;

		parents[k]->w_means[inParentsListIDs[k]] = parents[k]->w_variates[inParentsListIDs[k]];
	}

	float decay = std::max(expf(-certaintyDecay * powf(b_variate - b_mean, 2.0f) * b_precision), .8f);
	//float decay = std::max(expf(-certaintyDecay * abs(b_variate - b_mean) * sqrtf(b_precision)), .8f);

	b_precision = b_precision * decay + observationImportance;

	b_mean = b_variate;
}




void ANode::setActivation(float newX)
{
	float deltaFX = std::clamp(newX, .0f, 1.f) - fx;
	fx += deltaFX;
	x = newX;
	epsilon = x - mu;

	for (int k = 0; k < nChildren; k++)
	{
		children[k]->mu += deltaFX * w_variates[k];
		children[k]->epsilon -= deltaFX * w_variates[k];
	}
}

void ANode::prepareToReceivePredictions()
{
	mu = b_variate;
}

void ANode::transmitPredictions()
{
	for (int k = 0; k < nChildren; k++)
	{
		children[k]->mu += fx * w_variates[k];
		children[k]->epsilon -= fx * w_variates[k];
	}
}
