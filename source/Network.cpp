#include "Network.h"	



Network::Network(int _datapointSize, int _labelSize) :
	datapointSize(_datapointSize), labelSize(_labelSize)
{

	output = new float[labelSize];

	groups.push_back(new Group(datapointSize, 0));
	groups.push_back(new Group(labelSize, 1));

	groupOffsets.push_back(0);
	groupOffsets.push_back(datapointSize);


	int initialnNodes = datapointSize + labelSize;
	nodes.resize(initialnNodes);


	for (int i = 0; i < datapointSize; i++)
	{
		nodes[i] = new Node(groups[0]);
	}
	for (int i = datapointSize; i < initialnNodes; i++)
	{
		nodes[i] = new Node(groups[1]);
	}


	for (int j = 0; j < initialnNodes; j++) {
		nodes[j]->localXReg = 0.f; // no regularisation for the observations. (label and datapoint)
	}


	isInitialized = false;
	learningMode = false;
	testingMode = false;
}


Network::~Network() 
{
	delete[] output;

	for (int i = 0; i < nodes.size(); i++)
	{
		delete nodes[i];
	}

	for (int i = 0; i < groups.size(); i++)
	{
		delete groups[i];
	}
}


void Network::initialize()
{
	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->prepareToReceivePredictions();
	}
	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->transmitPredictions();
	}


	for (int i = 0; i < groups.size(); i++)
	{
#ifdef FREE_NODES
		bool _freeGroup = false;
		if ((int) groups[i]->childrenGroups.size() == 0) _freeGroup = true;
#endif

		float _sumEps2 = .0f;
		for (int j = groupOffsets[i]; j < groupOffsets[i] + groups[i]->nNodes; j++)
		{
			nodes[j]->epsilon = nodes[j]->x - nodes[j]->mu;

#ifdef FREE_NODES
			if (_freeGroup) {
				nodes[j]->isFree = true;
				nodes[j]->x = nodes[j]->mu;
				nodes[j]->epsilon = .0f;
			}
#endif

			groups[i]->newSumEps2 += nodes[j]->epsilon * nodes[j]->epsilon;
		}

		groups[i]->onSumEps2Recomputed();
	}

#ifdef FREE_NODES // the "observation" group is never free in our setting.
	for (int j = groupOffsets[0]; j < groupOffsets[0] + groups[0]->nNodes; j++)
	{
		nodes[j]->isFree = false;
	}
#endif

	isInitialized = true;
}



void Network::learn(float* _datapoint, float* _label, int nSteps)
{

	if (!isInitialized)
	{
		LOG("Network must be initialized before learning. Call initialize()");
		return;
	}

	if (!learningMode)
	{
		LOG("Network must be switched in learning mode before learning. Call readyForLearning()");
		return;
	}

	setActivities(_datapoint, _label);

	int nClamped = datapointSize + labelSize;

	//float previousEnergy = computeTotalActivationEnergy();
	for (int s = 0; s < nSteps; s++) 
	{
		//LOG(previousEnergy);

		std::shuffle(permutation.begin(), permutation.end(), generator);
		for (int i = 0; i < nodes.size() - nClamped; i++)
		{
#ifdef ANALYTICAL_X
			nodes[permutation[i]]->analyticalXUpdate();
#else
			nodes[permutation[i]]->XGradientStep();
#endif 
		}

#ifndef ASYNCHRONOUS_UPDATES
		for (int i = 0; i < nodes.size(); i++)
		{
			nodes[i]->prepareToReceivePredictions();
		}
		for (int i = 0; i < nodes.size(); i++)
		{
			nodes[i]->transmitPredictions();
		}
		for (int i = 0; i < nodes.size(); i++)
		{
			nodes[i]->epsilon = nodes[i]->x - nodes[i]->mu;
		}
#endif

		//float currentEnergy = computeTotalActivationEnergy();
		//previousEnergy = currentEnergy;
	}
	//LOG(previousEnergy << "\n\n");


#ifdef VANILLA_PREDICTIVE_CODING 
	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->predictiveCodingWxGradientStep();
	}
#else

	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->setAnalyticalWX();
	}
	
	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->calcifyWB();
	}
#endif
}

void Network::evaluate(float* _datapoint, int nSteps) 
{

	if (!isInitialized)
	{
		LOG("Network must be initialized before being tested. Call initialize()");
		return;
	}

	if (!testingMode)
	{
		LOG("Network must be switched in testing mode before testing. Call readyForTesting()");
		return;
	}

	setActivities(_datapoint, nullptr);


	int nClamped = datapointSize;
	

	//float previousEnergy = computeTotalActivationEnergy();
	for (int s = 0; s < nSteps; s++)
	{
		//LOG(previousEnergy);


		std::shuffle(permutation.begin(), permutation.end(), generator);
		for (int i = 0; i < nodes.size() - nClamped; i++)
		{
#ifdef ANALYTICAL_X
			nodes[permutation[i]]->analyticalXUpdate();
#else
			nodes[permutation[i]]->XGradientStep();
#endif 
		}

	

#ifndef ASYNCHRONOUS_UPDATES
		for (int i = 0; i < nodes.size(); i++)
		{
			nodes[i]->prepareToReceivePredictions();
		}
		for (int i = 0; i < nodes.size(); i++)
		{
			nodes[i]->transmitPredictions();
		}
		for (int i = 0; i < nodes.size(); i++)
		{
			nodes[i]->epsilon = nodes[i]->x - nodes[i]->mu;
		}
#endif

		//float currentEnergy = computeTotalActivationEnergy();
		//previousEnergy = currentEnergy;
	}
	//LOG(previousEnergy << "\n");

	for (int i = 0; i < labelSize; i++)
	{
		output[i] = nodes[datapointSize + i]->x;
	}
}



float Network::computeTotalActivationEnergy()
{
	float E2 = 0.f;

	for (int i = 0; i < nodes.size(); i++)
	{
		E2 += nodes[i]->epsilon * nodes[i]->epsilon; 
	}

	return E2;
}


void Network::setActivities(float* _datapoint, float* _label)
{
	std::vector<int> changedGroups(groups.size()); // The groups whose epsilons are changed by this function.
	std::fill(changedGroups.begin(), changedGroups.end(), 0);

	if (_datapoint != nullptr)
	{
		for (int i = 0; i < datapointSize; i++)
		{
			nodes[i]->setActivation(_datapoint[i]);
		}
		changedGroups[0] = 1;
		for (Group* g : groups[0]->childrenGroups) {
			changedGroups[g->id] = 1;
		}
	}
	if (_label != nullptr)
	{
		for (int i = 0; i < labelSize; i++)
		{
			nodes[i + datapointSize]->setActivation(_label[i]);
		}
		changedGroups[1] = 1;
		for (Group* g : groups[1]->childrenGroups) {
			changedGroups[g->id] = 1;
		}
	}

	
	for (int i = 0; i < groups.size(); i++)
	{
		if (changedGroups[i] == 0) continue;

		
		for (int j = groupOffsets[i]; j < groupOffsets[i] + groups[i]->nNodes; j++) 
		{
#ifdef FREE_NODES
			if (nodes[i]->isFree) [[unlikely]] {nodes[i]->x = nodes[i]->mu; }
#endif
			nodes[i]->epsilon = nodes[i]->x - nodes[i]->mu;

			groups[i]->newSumEps2 += nodes[i]->epsilon * nodes[i]->epsilon;
		}

		groups[i]->onSumEps2Recomputed();
	}


}



void Network::addGroup(int nNodes)
{
	groupOffsets.push_back(groupOffsets.back() + (int)groups.back()->nNodes);
	groups.push_back(new Group(nNodes, (int)groups.size()));

	nodes.resize(nodes.size() + nNodes);
	for (int i = groupOffsets.back(); i < groupOffsets.back() + nNodes; i++)
	{
		nodes[i] = new Node(groups.back());
	}

	isInitialized = false;
}

void Network::addConnexion(int originGroup, int destinationGroup)
{
	int nC = (int) groups[destinationGroup]->nNodes;
	int nP = (int) groups[originGroup]->nNodes;

	groups[originGroup]->childrenGroups.push_back(groups[destinationGroup]);

	Node** children = nodes.data() + groupOffsets[destinationGroup];
	Node** parents = nodes.data() + groupOffsets[originGroup];



	constexpr bool allowSelfConnexion = false;
	bool specialCase = ((originGroup == destinationGroup) && !allowSelfConnexion);


	int parentsOriginalNchildren = (int) parents[0]->children.size();
	std::vector<int> inParentIDs(nP);

	for (int i = 0; i < nC; i++)
	{
		std::fill(inParentIDs.begin(), inParentIDs.end(), parentsOriginalNchildren + i);

		children[i]->addParents(parents, inParentIDs.data(), nP);
	}
	
	for (int i = 0; i < nP; i++)
	{
		parents[i]->addChildren(children, nC, (specialCase ? i : -1));
	}

	isInitialized = false;
}


void Network::readyForLearning()
{
	if (!isInitialized) 
	{
		LOG("Network must be initialized before getting ready to learn. Call initialize()");
		return;
	}


#ifdef FREE_NODES
	for (int j = groupOffsets[1]; j < groupOffsets[1] + groups[1]->nNodes; j++)
	{
		nodes[j]->isFree = false;
	}
#endif


	int nClamped = datapointSize + labelSize;
	permutation.resize(nodes.size() - nClamped);
	for (int i = nClamped; i < nodes.size(); i++) permutation[i - nClamped] = i;

	learningMode = true;
	testingMode = false;
};

void Network::readyForTesting()
{

	if (!isInitialized)
	{
		LOG("Network must be initialized before getting ready to learn. Call initialize()");
		return;
	}


#ifdef FREE_NODES
	for (int j = groupOffsets[1]; j < groupOffsets[1] + groups[1]->nNodes; j++)
	{
		nodes[j]->isFree = true;
		nodes[j]->x = nodes[j]->mu;
		nodes[j]->epsilon = .0f;
	}
#endif

	int nClamped = datapointSize;
	permutation.resize(nodes.size() - nClamped);
	for (int i = nClamped; i < nodes.size(); i++) permutation[i - nClamped] = i;

	learningMode = false;
	testingMode = true;
};
