#include "Node.h"
#include <algorithm>
#include <limits>

// Recommended values:

float Node::xlr = 1.f;

float Node::wxPriorStrength = 1.f;

float Node::observationImportance = 1.f;
float Node::certaintyDecay = .01f;

float Node::xReg = .0f;
float Node::wxReg = .0f;

// bounds for analytical X update
constexpr float a = -1.f;
constexpr float b = 1.f;


Node::Node()
{
	isFree = false;

	parents.resize(0); // for completeness
	inParentsListIDs.resize(0); // for completeness

	children.resize(0);
	
	bx_mean = NORMAL_01 * .01f;
	bx_precision = wxPriorStrength;
	bx_variate = bx_mean;


	mu = bx_mean;
	x = mu;
	fx = F(x);

	tau = 1.0f;
	compute_sw();

	localXReg = xReg;

	computeLocalQuantities();
}



void Node::addParents(Node** newParents, int* newInParentIDs, int nNewParents)
{
	parents.insert(parents.end(), newParents, newParents + nNewParents);
	inParentsListIDs.insert(inParentsListIDs.end(), newInParentIDs, newInParentIDs + nNewParents);
}

void Node::addChildren(Node** newChildren, int nNewChildren, int specialCase)
{
	children.insert(children.end(), newChildren, newChildren + nNewChildren);

	wx_variates.resize(children.size());
	wx_means.resize(children.size());
	wx_precisions.resize(children.size());

	float amplitude = .01f;
	//float amplitude = .1f / sqrtf((float)(1 + (int)children.size()));

	for (int i = (int)children.size() - nNewChildren; i < children.size(); i++)
	{
		wx_variates[i] = amplitude * NORMAL_01;
		wx_means[i] = wx_variates[i];
		wx_precisions[i] = wxPriorStrength;
	}

	if (specialCase != -1) // if connecting to the same group and self connexions not allowed
	{
		wx_variates[specialCase] = .0f;
		wx_means[specialCase] = wx_variates[specialCase];
		wx_precisions[specialCase] = 1000000000.f; // TODO decay will lower this to the same value as other nodes after some time...
	}
}



void Node::XGradientStep() 
{
	float grad = (epsilon + x * localXReg) * tau;


#ifdef TANH
	float fprime = 1.f - fx * fx;
#elif defined(QSIGMOIDE)
	float fprime = .5f - 2.f * powf(fx - .5f, 2.f);
#elif defined(ID)
	float fprime = 1.0f;
#endif


	float grad_acc = .0f;

	float H = tau * (1.f + localXReg);
	
	for (int k = 0; k < children.size(); k++)
	{
		Node& c = *children[k];

		grad_acc += - c.tau * c.epsilon * wx_variates[k];
		float fpw = wx_variates[k] * fprime;


#ifdef TANH
		float hk = c.tau * fpw * (fpw + 2 * c.epsilon * fx);
#elif defined(QSIGMOIDE)
		float hk = c.tau * fpw * (fpw + 4 * c.epsilon * (fx - .5f));
#elif defined(ID)
		float hk = c.tau * fpw * fpw;
#endif

		
#ifdef NO_SECOND_ORDER
		// no operation needed
#elif defined SECOND_ORDER_TAU
		// no operation needed
#elif defined SECOND_ORDER_MAX
		H = std::max(H, abs(hk));
#elif defined SECOND_ORDER_L1
		H += abs(hk);
#endif
	}
	grad += fprime * grad_acc;

#ifdef NO_SECOND_ORDER
	H = 1.f;
#endif

	x -= xlr * grad / H;

	float deltaFX = F(x) - fx;
	fx += deltaFX;

#ifdef ASYNCHRONOUS_UPDATES
	epsilon = x - mu;

	for (int k = 0; k < children.size(); k++)
	{
		Node& c = *children[k];

		c.mu += deltaFX * wx_variates[k];

#ifdef FREE_NODES
		if (c.isFree) [[unlikely]] {c.x = c.mu; }
#endif

		c.computeLocalQuantities();
	}
#endif 
}


void Node::analyticalXUpdate()
{
	float stvw = .0f, stw2=.0f;
	for (int i = 0; i < children.size(); i++)
	{
		float tiwi = wx_variates[i] * children[i]->tau;
		stw2 += tiwi * wx_variates[i];
		float vi = children[i]->epsilon + wx_variates[i] * fx;
		stvw += tiwi * vi;
	}

#ifdef REGXL1
	float xstar = (tau * mu + stvw) / (tau + stw2 + localXReg);
	xstar = F(xstar);
	float Estar = tau * powf(xstar - mu, 2.0f) + xstar * (xstar * stw2 - 2.f * stvw) + localXReg; // +stv2
	float E0 = tau * mu * mu; // +stv2
	float bonus = .05f;
	xstar = (E0 - bonus < Estar) ? 0.f : xstar;
#else
	float xstar = (tau * mu + stvw) / (tau + stw2 + localXReg);
	xstar = F(xstar);
#endif

	
	x += (xstar - x) * xlr;


	float deltaFX = F(x) - fx;
	fx += deltaFX;

#ifdef ASYNCHRONOUS_UPDATES
	epsilon = x - mu;

	for (int k = 0; k < children.size(); k++)
	{
		Node& c = *children[k];

		c.mu += deltaFX * wx_variates[k];

#ifdef FREE_NODES
		if (c.isFree) [[unlikely]] {c.x = c.mu; }
#endif
		
		c.computeLocalQuantities();
	}
#endif 
}


void Node::setAnalyticalWX()
{
	float s1 = 1.0f / bx_precision;
	float s2 = bx_mean;


	for (int i = 0; i < parents.size(); i++)
	{
		int id = inParentsListIDs[i];
		float fi = parents[i]->fx;
		float t1 = fi / (parents[i]->wx_precisions[id] + wxReg * REGWX);

		s1 += fi * t1;
		s2 += t1 * parents[i]->wx_means[id] * parents[i]->wx_precisions[id];
	}

	// Either tau = 1, and this does nothing, or tau is computed in compute_sw, in which case it performs worse with this.
	//s1 *= tau;



	epsilon = (x - s2) / (1.f + s1);
	mu = x - epsilon;

	float sw2 = .0f;

	bx_variate = epsilon/ bx_precision + bx_mean;
	for (int i = 0; i < parents.size(); i++)
	{
		int id = inParentsListIDs[i];
		float fi = parents[i]->fx;
		float tau_i = parents[i]->wx_precisions[id];


		parents[i]->wx_variates[id] = (epsilon * fi + tau_i * parents[i]->wx_means[id])/(tau_i + wxReg * REGWX);


		sw2 += powf(parents[i]->wx_variates[id], 2.0f);
	}

	//sw2 = 1.f * powf(sw2, -.5f);
	//mu = bx_variate;
	//for (int i = 0; i < parents.size(); i++)
	//{
	//	int id = inParentsListIDs[i];
	//	parents[i]->wx_variates[id] *= sw2;
	//	mu += parents[i]->wx_variates[id] * parents[i]->fx;
	//}
	//epsilon = x - mu;
}


void Node::calcifyWB()
{
	for (int k = 0; k < children.size(); k++)
	{
		wx_precisions[k] += fx * fx * (- wx_precisions[k] * certaintyDecay + observationImportance) * children[k]->tau; //   * children[k]->tau
		wx_means[k] = wx_variates[k];
	}

	bx_precision += (- bx_precision * certaintyDecay + observationImportance) * tau; // tau *
	bx_mean = bx_variate;
}


void Node::predictiveCodingWxGradientStep()
{
	const float wlr = .1f;

	for (int k = 0; k < children.size(); k++)
	{
		wx_variates[k] -= wlr * fx * -children[k]->epsilon * children[k]->tau + wxReg * wx_variates[k];
	}

	bx_variate -= wlr * -epsilon * tau;
}


void Node::setActivation(float newX)
{
	x = newX;
	epsilon = x - mu;

	float deltaFX = F(x) - fx;
	fx += deltaFX;

	for (int k = 0; k < children.size(); k++)
	{
		children[k]->mu += deltaFX * wx_variates[k];

#ifdef FREE_NODES
		if (children[k]->isFree) [[unlikely]] {children[k]->x = children[k]->mu; }
#endif
	}
}

void Node::prepareToReceivePredictions()
{
	mu = bx_variate;
}

void Node::transmitPredictions()
{
	for (int k = 0; k < children.size(); k++)
	{
		children[k]->mu += fx * wx_variates[k];
	}
}

void Node::computeLocalQuantities() 
{
#ifdef FREE_NODES
	if (isFree) [[unlikely]] {x = mu;}
#endif
	
	epsilon = x - mu;
}



void Node::compute_sw() 
{
	//tau = children.size() == 0 ? 1.0f : 0.0f;
	//for (int i = 0; i < children.size(); i++) tau += powf(wx_variates[i] * children[i]->tau, 2.0f);
	//tau = sqrtf(tau);
	
	//tau = std::min(tau, 1.0f);
}

