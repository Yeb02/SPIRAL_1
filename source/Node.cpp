#include "Node.h"
#include <algorithm>
#include <limits>

// Recommended values:

float Node::xlr = .8f;

float Node::wxPriorStrength = .2f;

float Node::observationImportance = 1.0f;
float Node::certaintyDecay = .02f;

float Node::xReg = .01f;
float Node::wxReg = .01f;

// bounds for analytical X update
constexpr float a = -1.0f;
constexpr float b = 1.0f;


Node::Node(int _nChildren, Node** _children, int _nCoParents)
{
	isFree = false;

	parents.resize(0); // for completeness
	inParentsListIDs.resize(0); // for completeness

	children.resize(_nChildren);
	std::copy(_children, _children + _nChildren, children.data());

	float initialAmplitude = 1.f / sqrtf((float)(1 + _nCoParents));


	bx_mean = NORMAL_01 * .01f;
	bx_precision = wxPriorStrength;
	bx_variate = bx_mean;

	wx_means.resize(_nChildren);
	wx_precisions.resize(_nChildren);
	wx_variates.resize(_nChildren);

	std::fill(wx_precisions.begin(), wx_precisions.end(), wxPriorStrength);
	for (int i = 0; i < _nChildren; i++) wx_means[i] = NORMAL_01 * initialAmplitude;
	wx_variates.assign(wx_means.begin(), wx_means.end());


	mu = bx_mean;
	x = bx_mean;
	fx = F(x);

	tau = 1.0f;
	compute_sw();

	localXReg = xReg;

	computeLocalQuantities();
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


	bx_variate = epsilon/ bx_precision + bx_mean;
	for (int i = 0; i < parents.size(); i++)
	{
		int id = inParentsListIDs[i];
		float fi = parents[i]->fx;
		float tau_i = parents[i]->wx_precisions[id];

		parents[i]->wx_variates[id] = (epsilon * fi + tau_i * parents[i]->wx_means[id])/(tau_i + wxReg * REGWX);
	}
}


void Node::calcifyWB()
{
	compute_sw();

	for (int k = 0; k < children.size(); k++)
	{
		wx_precisions[k] = wx_precisions[k] * (1.0f - fx*fx*certaintyDecay) + observationImportance * fx * fx;
		wx_means[k] = wx_variates[k];
	}
	
	bx_precision = bx_precision * (1.0f - certaintyDecay) + observationImportance;
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
	tau = children.size() == 0 ? 1.0f : 0.0f;
	for (int i = 0; i < children.size(); i++) tau += powf(wx_variates[i] * children[i]->tau, 2.0f);
	tau = sqrtf(tau);

	//tau = 1.0f;
}

