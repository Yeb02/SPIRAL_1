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




Node::Node(Group* _group) :
	group(_group)
{
	// for completeness
	parents.resize(0); 
	inParentsListIDs.resize(0); 
	children.resize(0);
	wx_variates.resize(0);
	wx_precisions.resize(0);
	wx_means.resize(0);



	bx_mean = NORMAL_01 * .01f;

	bx_precision = .01f;
	//bx_precision = wxPriorStrength; 
	bx_variate = bx_mean;


	mu = bx_mean;
	x = mu;
	fx = F(x);

	localXReg = xReg;
#ifdef FREE_NODES
	isFree = false;
#endif
#ifdef ADVANCED_W_IMPORTANCE
	factor = .0f;
#endif
	epsilon = x - mu;
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

#ifdef INDIRECT_DESCENT
	prescribedXs.resize(children.size());
#endif

	// TODO unfortunately initialization still has a strong influence on performance.
	// Check that it is still the case in later versions of the algorithm.
	//float amplitude = .01f; 
	//float amplitude = .1f / sqrtf((float)(1 + (int)children.size()));
	float amplitude = 1.f / sqrtf((float)(children.size() * children.back()->parents.size()));

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
	float grad = (epsilon + x * localXReg) * group->tau; // TODO  xReg is better factored in the tau* (at least makes more sense) 
	//float grad = epsilon * group->tau + x * localXReg; 


#ifdef TANH
	float fprime = 1.f - fx * fx;
#elif defined(QSIGMOIDE)
	float fprime = .5f - 2.f * powf(fx - .5f, 2.f);
#elif defined(ID)
	float fprime = 1.0f;
#endif


	float grad_acc = .0f;

	float H = group->tau * (1.f + localXReg);
	
	for (int k = 0; k < children.size(); k++)
	{
		Node& c = *children[k];

		// Because of the current signification of tau, the children's taus are ignored
		grad_acc += -c.epsilon * wx_variates[k]; // *c.group->tau;
		float fpw = wx_variates[k] * fprime;


#ifdef TANH
		float hk = fpw * (fpw + 2 * c.epsilon * fx); // *c.group->tau;
#elif defined(QSIGMOIDE)
		float hk = fpw * (fpw + 4 * c.epsilon * (fx - .5f)); //* c.group->tau;
#elif defined(ID)
		float hk = fpw * fpw; //* c.group->tau;
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

		c.epsilon = c.x - c.mu;
	}
#endif 
}


void Node::analyticalXUpdate()
{
	float stvw = .0f, stw2=.0f;
	for (int i = 0; i < children.size(); i++)
	{
		float tiwi = wx_variates[i];  //*children[i]->group->tau; Because of the current signification of tau, the children's taus are ignored
		stw2 += tiwi * wx_variates[i];
		float vi = children[i]->epsilon + wx_variates[i] * fx;
		stvw += tiwi * vi;
	}

	
#ifdef XREG_IN_W
	float xstar = (group->tau * mu + stvw) / (stw2 + group->tau);
#else
	// xReg is better factored in the tau * (at least makes more sense) 
	float xstar = (group->tau * mu + stvw) / (stw2 + group->tau * (1.0f + localXReg));
#endif



#ifdef LEAST_ACTION
	//constexpr float gamma = .0f; // .5 * gamma actually 
	//constexpr float gamma = .3f; // .5 * gamma actually 
	constexpr float gamma = .2f; // .5 * gamma actually 

	float R = .0f;
	for (int i = 0; i < children.size(); i++)
	{
		//R += powf(children[i]->epsilon * wx_precisions[i], 2.0f); // alpha = 1
		//R += powf(children[i]->epsilon, 2.0f);				// alpha = 0
		R += powf(children[i]->epsilon, 2.0f) / wx_precisions[i]; // alpha = -1
	}
	R *= gamma;

	float E = .0f;
	for (int j = 0; j < parents.size(); j++)
	{
		//E += powf(parents[j]->fx * parents[j]->wx_precisions[inParentsListIDs[j]], 2.0f);  // alpha = 1
		//E += powf(parents[j]->fx, 2.0f);												// alpha = 0
		E += powf(parents[j]->fx, 2.0f) / parents[j]->wx_precisions[inParentsListIDs[j]]; // alpha = -1
	}
	E *= gamma;


	xstar = ((group->tau + E) * mu + stvw) / (stw2 + E + group->tau * (1.0f + localXReg) + R);
	//xstar = ((group->tau + E) * mu + stvw) / (stw2 + E + group->tau * (1.0f + localXReg + R));
	//xstar = (group->tau * (1.f + E) * mu + stvw) / (stw2 + group->tau * (E + 1.0f + localXReg + R));
#endif



	xstar = F(xstar);

	
	x += (xstar - x) * xlr;

	float deltaFX = F(x) - fx; // redundant
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
		
		c.epsilon = c.x - c.mu;
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

	

	// TODO should probably be commented out with the current use of tau
	//s1 *= group->tau;

#ifdef XREG_IN_W
	epsilon = ((localXReg * s1 + 1.f) * x - s2) / (1.f + (observationImportance + localXReg) * s1);
#else
	epsilon = (x - s2) / (1.f + s1 * observationImportance);
#endif
	mu = x - epsilon;


	bx_variate = observationImportance * epsilon/ bx_precision + bx_mean;
	for (int i = 0; i < parents.size(); i++)
	{
		int id = inParentsListIDs[i];
		float fi = parents[i]->fx;
		float tau_i = parents[i]->wx_precisions[id];

#ifdef XREG_IN_W
		parents[i]->wx_variates[id] = (observationImportance * epsilon * fi + tau_i * parents[i]->wx_means[id] - localXReg * mu * fi)
			/ (tau_i + wxReg * REGWX);
#else 
		parents[i]->wx_variates[id] = (observationImportance * epsilon * fi + tau_i * parents[i]->wx_means[id]) / (tau_i + wxReg * REGWX);
#endif
		
	}
}


void Node::calcifyWB()
{


	for (int k = 0; k < children.size(); k++)
	{
#ifdef ADVANCED_W_IMPORTANCE
		float ff = powf(fx * children[k]->factor, 2.0f);
		wx_precisions[k] += -wx_precisions[k] * certaintyDecay + ff * observationImportance;
		//wx_precisions[k] += ff * (-wx_precisions[k] * certaintyDecay + observationImportance);
#else
		//wx_precisions[k] +=  -wx_precisions[k] * certaintyDecay + fx * fx * observationImportance; // requires a smaller decay, 1e-4
		wx_precisions[k] += fx * fx * (-wx_precisions[k] * certaintyDecay + observationImportance); // requires a larger decay, 1e-3
#endif
		wx_means[k] = wx_variates[k];
	}

#ifdef ADVANCED_W_IMPORTANCE
	float ff = powf(factor, 2.0f);
	bx_precision += -bx_precision * certaintyDecay + ff * observationImportance;
	//bx_precision += ff * (-bx_precision * certaintyDecay + observationImportance);
#else
	bx_precision += -bx_precision * certaintyDecay + observationImportance; 
#endif
	bx_mean = bx_variate;
}


void Node::predictiveCodingWxGradientStep()
{
	const float wlr = .1f;

	for (int k = 0; k < children.size(); k++)
	{
		wx_variates[k] -= wlr * (fx * -children[k]->epsilon + wxReg * wx_variates[k]);
	}

	bx_variate -= wlr * -epsilon;
}


void Node::setActivation(float newX)
{
	x = newX;
	epsilon = x - mu; // the group's tracking of epsilons is handled by the only function that calls this one, Network::setActivities .

	float deltaFX = F(x) - fx;
	fx += deltaFX;

	for (int k = 0; k < children.size(); k++)
	{
		children[k]->mu += deltaFX * wx_variates[k];
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



#ifdef INDIRECT_DESCENT

constexpr float lambdaX = .0f;  // LOCAL x reg TODO !
constexpr float lambdaXprime = .0f;

void Node::computeOptimalXs()
{
	if (parents.size() == 0) return;

	float _a = .0f;
	float _b = .0f;

	for (int i = 0; i < parents.size(); i++)
	{
		int id = inParentsListIDs[i];
		float wi = parents[i]->wx_variates[id];

		float par_tau = parents[i]->loco_tau - wi * wi;

		float wotl = wi / (lambdaXprime + par_tau);

		float par_mu = (parents[i]->loco_mu - (epsilon + wi * parents[i]->fx)) / par_tau;

		_a += wi*wotl;
		_b += par_mu*wotl;
	}


	
	float temp_epsilon = ((lambdaX * _a + 1.f) * x - _b) / ((1.f + lambdaX) * _a + 1.f);
	float temp_mu = x - temp_epsilon;

	for (int i = 0; i < parents.size(); i++)
	{
		int id = inParentsListIDs[i];
		float wi = parents[i]->wx_variates[id];
		float mui = parents[i]->mu;

		float par_tau = parents[i]->loco_tau - wi * wi;
		float par_mu = (parents[i]->loco_mu - (epsilon + wi * parents[i]->fx)) / par_tau;

		parents[i]->prescribedXs[id] = (temp_epsilon * wi + par_tau * par_mu - lambdaX * temp_mu * wi) / (par_tau + lambdaXprime);
	}
}


void Node::setXToBarycentre()
{
	if (children.size() == 0) {
		analyticalXUpdate();
		return;
	}


	float _baryX = .0f;
	float _div = 0.f;
	for (int i = 0; i < children.size(); i++)
	{
		//float _f = abs(wx_variates[i]);
		float _f = powf(wx_variates[i], 2.f);
		_baryX += prescribedXs[i] * _f;
		_div += _f;
	}
	x = F(_baryX / _div);
	fx = x;
}

void Node::computeLocos()
{
	if (children.size() == 0) {
		
		loco_tau = 1.f + lambdaXprime;
		loco_mu = mu;

		return;
	}

	float svw = .0f, sw2 = .0f;
	for (int i = 0; i < children.size(); i++)
	{
		float wi = wx_variates[i];  
		sw2 += wi * wi;
		float vi = children[i]->epsilon + wi * fx;
		svw += wi * vi;
	}
	
	loco_tau = 1.f + lambdaXprime + sw2;
	loco_mu = mu + svw; // actually =loco_mu*loco_tau, because the term induced by the i-th child needs be easily removed by this child in its computeOptimalXs
}
#endif