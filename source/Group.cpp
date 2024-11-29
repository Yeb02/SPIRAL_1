#include "Group.h"


Group::Group(int _nNodes, int _id) :
	id(_id)
{
	nNodes = (float)_nNodes;

	sumEps2 = .0f;
	tau = 1.0f;

	newSumEps2 = .0f; 
};

constexpr float tauMin = .001f;
constexpr float tauMax = 1000.0f;
//constexpr float tauMin = 1.f;
//constexpr float tauMax = 1.f;

void Group::onOneEpsUpdated(float oldEps, float newEps)
{
	float _delta = newEps * newEps - oldEps * oldEps;
	sumEps2 += _delta;

	tau = std::clamp(sumEps2 / nNodes, tauMin, tauMax);
};


void Group::onSumEps2Recomputed()
{
	float _delta = newSumEps2 - sumEps2;
	sumEps2 = newSumEps2;

	tau = std::clamp(sumEps2 / nNodes, tauMin, tauMax);

	newSumEps2 = .0f;
}
