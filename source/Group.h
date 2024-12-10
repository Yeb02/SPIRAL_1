#pragma once

#include "SPIRAL_includes.h"


struct Group 
{
	int id; // The position of this group in the parent Network "groups" vector. 

	std::vector<Group*> childrenGroups;

	float nNodes; // a float to avoid casting from int in all computations. Is set at construction and never changed afterwards.
	float sumEps2; // sum(epsilon^2) over nodes in this group
	float avgEps2; // sum(epsilon^2)/nNodes 


	float logTau; 
	float tau; 
	float age;


	Group(int _nNodes, int _id);

	void updateTau();
};