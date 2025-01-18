#include "Group.h"


constexpr float tauMin = .001f;
constexpr float tauMax = 1000.0f;
//constexpr float tauMin = 1.f;
//constexpr float tauMax = 1.f;

Group::Group(int _nNodes, int _id) :
	id(_id)
{
	nNodes = (float)_nNodes;

	sumEps2 = -1.0f;
	avgEps2 = -1.0f;

	tau = 1.0f;
	logTau = .0f;
	age = 5.f;
};


void Group::updateTau()
{
	constexpr float decay = 1.f - .05f;


	avgEps2 = sumEps2 / nNodes;

	if (childrenGroups.size() == 0) return;

	float avgChildrenEps2 = .0f;
	float nChildren = .0f;
	for (int i = 0; i < childrenGroups.size(); i++) 
	{
		avgChildrenEps2 += childrenGroups[i]->sumEps2;
		nChildren += childrenGroups[i]->nNodes;
	}
	avgChildrenEps2 /= nChildren;

	logTau += logf(avgEps2 / avgChildrenEps2) / age;
	age = age * decay + 1.f;

	tau = expf(logTau);
}

