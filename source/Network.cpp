#include "Network.h"	



Network::Network(int _datapointSize, int _labelSize, int nLayers, int* sizes) :
	datapointSize(_datapointSize), labelSize(_labelSize)
{

	output = new float[labelSize];

	int nNodes = 0;
	for (int i = 0; i < nLayers; i++) {
		nNodes += sizes[i];
	}
	nodes.resize(nNodes);

	int offset = 0;
	for (int i = 0; i < nLayers; i++)
	{
		// Keep in mind that sizes[] is padded by zeros (see main.cpp)
		// Will throw an error if adress sanitizer is on though. But so much more convenient.
		Node** children = nodes.data() + offset - sizes[i - 1];

		for (int j = 0; j < sizes[i]; j++) {
			nodes[offset + j] = new Node(sizes[i - 1], children, sizes[i]);
		}

		for (int j = 0; j < sizes[i - 1]; j++) {
			children[j]->parents.resize(sizes[i]);
			std::copy(nodes.data() + offset,
				nodes.data() + offset + sizes[i],
				nodes[offset - sizes[i - 1] + j]->parents.data()
			);
			children[j]->inParentsListIDs.resize(sizes[i]);
			std::fill(children[j]->inParentsListIDs.begin(),
				children[j]->inParentsListIDs.end(),
				j
			);
		}

		offset += sizes[i];
	}

	// Set the nodes quantities to their correct values to prepare inference
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

	for (int j = 0; j < sizes[0]; j++) {
		nodes[j]->localXReg = 0.f; // no regularisation for the observations.
	}
	
}


Network::~Network() 
{
	delete[] output;
	for (int i = 0; i < nodes.size(); i++)
	{
		delete nodes[i];
	}
}


void Network::learn(float* _datapoint, float* _label, int nSteps)
{
	setActivities(_datapoint, _label);

	int nClamped = datapointSize + labelSize;

	std::vector<int> permutation(nodes.size() - nClamped);
	for (int i = nClamped; i < nodes.size(); i++) permutation[i - nClamped] = i;


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
	setActivities(_datapoint, nullptr);


	int nClamped = datapointSize;


	std::vector<int> permutation(nodes.size() - nClamped);
	for (int i = nClamped; i < nodes.size(); i++) permutation[i - nClamped] = i;

	

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



void Network::readyForLearning()
{
	for (int i = datapointSize; i < datapointSize + labelSize; i++) nodes[i]->isFree = false;
};

void Network::readyForTesting()
{
	for (int i = datapointSize; i < datapointSize + labelSize; i++)
	{
#ifdef FREE_NODES
		nodes[i]->isFree = true; 
#endif
		nodes[i]->x = nodes[i]->mu;
	}

};
