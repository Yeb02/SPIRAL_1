#include "Network.h"	



Network::Network(int _datapointSize, int _labelSize) :
	datapointSize(_datapointSize), labelSize(_labelSize)
{

	output = new float[labelSize];

	int initialnNodes = datapointSize + labelSize;
	nodes.resize(initialnNodes);
	for (int i = 0; i < initialnNodes; i++)
	{
		nodes[i] = new Node();
	}

	groupSizes.push_back(datapointSize);
	groupSizes.push_back(labelSize);

	groupOffsets.push_back(0);
	groupOffsets.push_back(datapointSize);

#ifdef FREE_NODES
	freeGroups.push_back(0);
	freeGroups.push_back(1);
#endif 

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
	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->computeLocalQuantities();
	}


#ifdef FREE_NODES
	for (int i = 0; i < groupSizes.size(); i++)
	{
		if (!freeGroups[i]) continue;
		for (int j = groupOffsets[i]; j < groupOffsets[i] + groupSizes[i]; j++)
		{
			nodes[j]->isFree = true;
			nodes[j]->x = nodes[j]->mu;
			nodes[j]->epsilon = .0f;
		}
	}
#endif

	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->compute_sw();
	}

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
			nodes[i]->computeLocalQuantities();
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
		
		if (i == 0) {
			float _aa = 0.0f;
		}
		nodes[i]->setAnalyticalWX();
	}
	

	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->calcifyWB();
	}

	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->compute_sw();
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
			nodes[i]->computeLocalQuantities();
		}
#endif

		//float currentEnergy = computeTotalActivationEnergy();
		//previousEnergy = currentEnergy;
	}
	//LOG(previousEnergy);

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

	if (_datapoint != nullptr)
	{
		for (int i = 0; i < datapointSize; i++)
		{
			nodes[i]->setActivation(_datapoint[i]);
		}
	}
	if (_label != nullptr)
	{
		for (int i = 0; i < labelSize; i++)
		{
			nodes[i + datapointSize]->setActivation(_label[i]);
		}
	}

	// inefficient, but no easy way to do much better
	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->computeLocalQuantities(); 
	}
}



void Network::addGroup(int nNodes)
{
	groupOffsets.push_back(groupOffsets.back() + groupSizes.back());
	groupSizes.push_back(nNodes);

	nodes.resize(nodes.size() + nNodes);
	for (int i = groupOffsets.back(); i < groupOffsets.back() + groupSizes.back(); i++)
	{
		nodes[i] = new Node();
	}

#ifdef FREE_NODES
	freeGroups.push_back(1);
#endif 

	isInitialized = false;
}

void Network::addConnexion(int originGroup, int destinationGroup)
{
	int nC = groupSizes[destinationGroup];
	int nP = groupSizes[originGroup];

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


#ifdef FREE_NODES
	freeGroups[originGroup] = 0;
#endif 
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
	for (int j = groupOffsets[1]; j < groupOffsets[1] + groupSizes[1]; j++)
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
	for (int j = groupOffsets[1]; j < groupOffsets[1] + groupSizes[1]; j++)
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
