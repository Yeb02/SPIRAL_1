#include "Node.h"

// Recommended values:

float Node::xlr = .8f;

float Node::wxPriorStrength = .2f;
float Node::wtPriorStrength = .2f;

float Node::observationImportance = 1.0f;
float Node::certaintyDecay = .02f;

float Node::energyDecay = .05f;
float Node::connexionEnergyThreshold = 10.f;

float Node::xReg = .01f;
float Node::wxReg = .01f;
float Node::wtReg = .01f;



Node::Node(int _nChildren, Node** _children)
{
	
	parents.resize(0); // for completeness
	inParentsListIDs.resize(0); // for completeness

	children.resize(_nChildren);
	std::copy(_children, _children + _nChildren, children.data());

	bx_mean = NORMAL_01 * .1f;
	bx_precision = wxPriorStrength;
	bx_variate = bx_mean;

	wx_means.resize(_nChildren);
	wx_precisions.resize(_nChildren);
	wx_variates.resize(_nChildren);

	std::fill(wx_precisions.begin(), wx_precisions.end(), wxPriorStrength);
	for (int i = 0; i < _nChildren; i++) wx_means[i] = NORMAL_01 * .1f;
	wx_variates.assign(wx_means.begin(), wx_means.end());

	mu = bx_mean;
	x = bx_mean;
	fx = F(x);

	connexionEnergies.resize(_nChildren);
	std::fill(connexionEnergies.begin(), connexionEnergies.end(), .0f);
	accumulatedEnergy = .0f;

#if defined(DYNAMIC_PRECISIONS) || defined(FIXED_PRECISIONS_BUT_CONTRIBUTE)
	leps = .0f;
#endif

#ifdef FIXED_PRECISIONS_BUT_CONTRIBUTE
	tau_mean = 0.0f;
	tau_precision = 1.0f;
#endif


#ifdef DYNAMIC_PRECISIONS

	bt_mean = .5f + NORMAL_01 * .01f; // -1 because typical values are in [-1, 1], so N(0, tau=exp(.5)) is better than N(0, 1)
	bt_precision = wtPriorStrength;
	bt_variate = bt_mean;


	wt_means.resize(_nChildren);
	wt_precisions.resize(_nChildren);
	wt_variates.resize(_nChildren);

	std::fill(wt_precisions.begin(), wt_precisions.end(), wtPriorStrength);
	for (int i = 0; i < _nChildren; i++) wt_means[i] = NORMAL_01 * .01f;
	wt_variates.assign(wt_means.begin(), wt_means.end());
	t = bt_mean;

#else
	tau = 1.0f;
#endif

	computeLocalQuantities();
}


void Node::XGradientStep() 
{
	float grad = (epsilon + x * xReg) * tau;
	//float grad = epsilon * tau + x * xReg; // and H
#ifdef TANH
	float fprime = 1.f - fx * fx;			   
#else // quasiSigmoide
	float fprime = .5f - 2.f * powf(fx - .5f, 2.f); 
#endif
	float grad_acc = .0f;

	float H = tau * (1.f + xReg);
	//float H = tau + xReg; // and grad
	
	for (int k = 0; k < children.size(); k++)
	{
		Node& c = *children[k];

		grad_acc += - c.tau * c.epsilon * wx_variates[k];

		float fpw = wx_variates[k] * fprime;
#ifdef TANH
		float hk = c.tau * fpw * (fpw + 2 * c.epsilon * fx);
#else // quasiSigmoide
		float hk = c.tau * fpw * (fpw + 4 * c.epsilon * (fx - .5f));
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

#ifdef DYNAMIC_PRECISIONS
		c.t += deltaFX * wt_variates[k];
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

#ifndef WBX_IGNORE_TAU
	s1 *= tau;
#endif

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


void Node::setAnalyticalWT()
{
#ifdef DYNAMIC_PRECISIONS
	float s1 = 1.0f / bt_precision;
	float s2 = bt_mean;


	for (int i = 0; i < parents.size(); i++)
	{
		int id = inParentsListIDs[i];
		float fi = parents[i]->fx;
		float t1 = fi / (parents[i]->wt_precisions[id] + wtReg * REGWT);

		s1 += fi * t1;
		s2 += t1 * parents[i]->wt_means[id] * parents[i]->wt_precisions[id];
	}

	float r = leps;
	if (children.size() != 0) {
		for (int i = 0; i < children.size(); i++)
		{
			r += children[i]->leps;
		}
	}
	float m = leps - r / (float) (children.size() +1);


	t = (s2 + m * s1) / (1.f + s1);
	tau = expf(t);

	float ambt = m - t;
	bx_variate = ambt / bx_precision + bx_mean;
	for (int i = 0; i < parents.size(); i++)
	{
		int id = inParentsListIDs[i];
		float fi = parents[i]->fx;
		float tau_i = parents[i]->wt_precisions[id];

		parents[i]->wt_variates[id] = (ambt*fi + tau_i * parents[i]->wt_means[id]) / (tau_i + wtReg * REGWT);
	}
#endif
}


void Node::calcifyWB()
{
	// Could dedicate a separate function to the topological operations, for clarity.

	accumulatedEnergy = accumulatedEnergy * (1.f - energyDecay) + epsilon * epsilon;


	for (int k = 0; k < children.size(); k++)
	{
		connexionEnergies[k] = connexionEnergies[k] * (1.0f - fx * fx * energyDecay) + wx_precisions[k] * powf(wx_means[k] - wx_variates[k], 2.0f);


		wx_precisions[k] = wx_precisions[k] * (1.0f - fx*fx*certaintyDecay) + observationImportance * fx * fx;
		wx_means[k] = wx_variates[k];
#ifdef DYNAMIC_PRECISIONS
		wt_precisions[k] = wt_precisions[k] * (1.0f - certaintyDecay) + observationImportance; // TODO test: do wt benefit from fx*fx ?
		wt_means[k] = wt_variates[k];
#endif
	}
	
	bx_precision = bx_precision * (1.0f - certaintyDecay) + observationImportance;
	bx_mean = bx_variate;
#ifdef DYNAMIC_PRECISIONS
	bt_precision = bt_precision * (1.0f - certaintyDecay) + observationImportance;
	bt_mean = bt_variate;
#endif



#ifdef FIXED_PRECISIONS_BUT_CONTRIBUTE
	float r = leps;
	if (children.size() != 0) {
		for (int i = 0; i < children.size(); i++)
		{
			r += children[i]->leps;
		}
}
	float m = leps - r / (float)(children.size() + 1);

	tau_mean = (tau_mean * tau_precision + m) / (tau_precision + 1.0f);
	tau_precision = tau_precision * (1.0f - certaintyDecay) + observationImportance;
	tau = expf(tau_mean);
#endif
}


void Node::pruneUnusedConnexions() 
{
	int nCm1 = (int)children.size() - 1;
	for (int i = 0; i < nCm1 + 1; i++)
	{
		if (connexionEnergies[i] > connexionEnergyThreshold)  // TODO think about the role of abs(w).
			[[unlikely]]
		{
			// A more efficient approach to deleting a small number of elements in a vector is to swap the deleted element 
			// with the last in the vector and then decrement the vector size. (Avoids costly reindicing and array moves)
			wx_variates[i] = wx_variates[nCm1];
			wx_means[i] = wx_means[nCm1];
			wx_precisions[i] = wx_precisions[nCm1];
			connexionEnergies[i] = connexionEnergies[nCm1];

#ifdef DYNAMIC_PRECISIONS
			wt_variates[i] = wt_variates[nCm1];
			wt_means[i] = wt_means[nCm1];
			wt_precisions[i] = wt_precisions[nCm1];
#endif

			// the child that will be moved to the i-th position in the children vector.
			for (int j = 0; j < children[nCm1]->parents.size(); i++)
			{
				if (children[nCm1]->parents[j] == this) [[unlikely]]
				{
					children[nCm1]->inParentsListIDs[j] = i;
					break;
				}
			}

			// the child that "this" is being disconnecting from
			int cinpm1 = (int)children[i]->parents.size() - 1;
			for (int j = 0; j < cinpm1+1; i++)
			{
				if (children[i]->parents[j] == this) [[unlikely]] 
				{
					children[i]->parents[j] = children[i]->parents[cinpm1];
					children[i]->parents.resize(cinpm1);
					children[i]->inParentsListIDs[j] = children[i]->inParentsListIDs[cinpm1];
					children[i]->inParentsListIDs.resize(cinpm1);
					break;
				}
			}

			children[i] = children[nCm1];
			i--;
			nCm1--;
		}
	}


	if (nCm1 + 1 != children.size()) [[unlikely]]
	{
		nCm1++; // = new nChildren.
		wx_variates.resize(nCm1);
		wx_means.resize(nCm1);
		wx_precisions.resize(nCm1);
		connexionEnergies.resize(nCm1);

#ifdef DYNAMIC_PRECISIONS
		wt_variates.resize(nCm1);
		wt_means.resize(nCm1);
		wt_precisions.resize(nCm1);
#endif

		LOG("\nRemoved");
		LOG((int)children.size() - nCm1);
		LOGL("existing connexions.");

		children.resize(nCm1);
	}
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

#ifdef DYNAMIC_PRECISIONS
		children[k]->t += deltaFX * wt_variates[k];
#endif
	}
}

void Node::prepareToReceivePredictions()
{
	mu = bx_variate;
#ifdef DYNAMIC_PRECISIONS
	t = bt_variate;
#endif
}

void Node::transmitPredictions()
{
	for (int k = 0; k < children.size(); k++)
	{
		children[k]->mu += fx * wx_variates[k];
#ifdef DYNAMIC_PRECISIONS
		children[k]->t += fx * wt_variates[k];
#endif
	}
}

void Node::computeLocalQuantities() 
{
	epsilon = x - mu;

#ifdef DYNAMIC_PRECISIONS
	tau = expf(t);
#endif
}

#if defined(DYNAMIC_PRECISIONS) || defined(FIXED_PRECISIONS_BUT_CONTRIBUTE)
void Node::computeLeps() {
	leps = logf(epsilon * epsilon + .00000001f);
}
#endif

