#pragma once

#include "SPIRAL_includes.h"


struct Group 
{
	int id; // The position of this group in the parent Network "groups" vector. 

	//std::vector<Group*> parentGroups; monitoring when grahs become more complex ?

	// used to make the update more efficient at inference. When x is updated, children's epsilons (mus) change as well.
	// therefore the children groups' "sumEps2" must be updated, and the coparents' (parents of the children of this group)
	// sumChildrenEps2 too. 
	// Also used #ifdef FREE_NODES: a group is free if it is not clamped and has no children.
	std::vector<Group*> childrenGroups;

	float nNodes; // a float to avoid casting from int in all computations. Is set at construction and never changed afterwards.
	float sumEps2; // sum(epsilon^2) over nodes in this group
	float tau; // = sumEps2 / nNodes


	// a buffer that accumulates the new sumEps2 resulting from a change of fx in a node of a parent group.
	// Used in the x update function(s) at inference for efficiency improvements.
	float newSumEps2; 

	Group(int _nNodes, int _id);


	// called when one of this group's node's epsilon changes, because it 
	// had its x -> epsilon updated during inference.
	void onOneEpsUpdated(float oldEps, float newEps);

	// functionnality partially handled to the parent network for efficiency
	void onSumEps2Recomputed();
};