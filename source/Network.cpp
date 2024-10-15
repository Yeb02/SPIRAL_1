#include "Network.h"	


float Network::KC = 1.f;
float Network::KN = 1.f;

Network::Network(int _datapointSize, int _labelSize, int nLayers, int* sizes) :
	datapointSize(_datapointSize), labelSize(_labelSize)
{
	output = new float[labelSize];

	if (sizes != nullptr) {
		dynamicTopology = false;
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
				nodes[offset + j] = new Node(sizes[i - 1], children);
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
	}
	else 
	{
		dynamicTopology = true;

		nodes.resize(datapointSize + labelSize);
		for (int i = 0; i < nodes.size(); i++)
		{
			nodes[i] = new Node(0, nullptr);
		}
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


	float previousEnergy = computeTotalActivationEnergy();
	for (int s = 0; s < nSteps; s++) 
	{
		LOG(previousEnergy);

		std::shuffle(permutation.begin(), permutation.end(), generator);
		for (int i = 0; i < nodes.size() - nClamped; i++)
		{
			if (permutation[i] == 794) {
				int a = 1;
			}
			nodes[permutation[i]]->XGradientStep();
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

		float currentEnergy = computeTotalActivationEnergy();
		previousEnergy = currentEnergy;
	}
	LOG(previousEnergy << "\n\n");


#if defined(DYNAMIC_PRECISIONS) || defined(FIXED_PRECISIONS_BUT_CONTRIBUTE)
	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->computeLeps();
	}
#endif

#ifdef VANILLA_PREDICTIVE_CODING 
	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->predictiveCodingWxGradientStep();
	}
#else
	for (int s = 0; s < 1; s++) // if DYNAMIC_PRECISIONS is enabled, it is not redundant to perform those operations more than once. TODO test.
	{
		for (int i = 0; i < nodes.size(); i++)
		{
			nodes[i]->setAnalyticalWX();
			nodes[i]->setAnalyticalWT();
		}
	}

	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->calcifyWB();
	}
#endif

	if (dynamicTopology) topologicalOperations();
}

void Network::evaluate(float* _datapoint, int nSteps) 
{
	setActivities(_datapoint, nullptr);


	int nClamped = datapointSize;

#ifdef RANDOM_UPDATE_ORDER
	std::vector<int> permutation(nodes.size() - nClamped);
	for (int i = nClamped; i < nodes.size(); i++) permutation[i - nClamped] = i;
#endif 
	

	float previousEnergy = computeTotalActivationEnergy();
	for (int s = 0; s < nSteps; s++)
	{
		LOG(previousEnergy);

#ifdef RANDOM_UPDATE_ORDER
		std::shuffle(permutation.begin(), permutation.end(), generator);
		for (int i = 0; i < nodes.size() - nClamped; i++)
		{
			nodes[permutation[i]]->XGradientStep();
		}
#else 
		for (int i = nClamped; i < nodes.size(); i++)
		{
			nodes[i]->XGradientStep();
		}
#endif
	

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

		float currentEnergy = computeTotalActivationEnergy();
		previousEnergy = currentEnergy;
	}
	LOG(previousEnergy);

	for (int i = 0; i < labelSize; i++)
	{
		output[i] = nodes[datapointSize + i]->x;
	}
}



float Network::computeTotalActivationEnergy()
{
	float E2 = 0.f;
	float Elog = 0.f;

	for (int i = 0; i < nodes.size(); i++)
	{
		E2 += nodes[i]->epsilon * nodes[i]->epsilon * nodes[i]->tau;
		Elog += - logf(nodes[i]->tau); 
		// + constants
	}

	return (E2 + Elog) * .5f;
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


void Network::topologicalOperations()
{
	nodesOverKC.clear();

	float acc = .0f;
	for (int i = 0; i < nodes.size(); i++)
	{
		if (nodes[i]->accumulatedEnergy > KC) {
			acc += nodes[i]->accumulatedEnergy - KC;
			nodesOverKC.push_back(nodes[i]);
		}

		nodes[i]->pruneUnusedConnexions();
	}

	if (acc > KN) {

		LOG("\nAdding a parent to");
		LOG((int)nodesOverKC.size());
		LOGL("existing children.");

		Node* newNode = new Node((int)nodesOverKC.size(), nodesOverKC.data());

		nodes.push_back(newNode);
		newNode->transmitPredictions();

		int a = 0;
		for (int i = 0; i < nodesOverKC.size(); i++)
		{
			a += (nodesOverKC[i]->children.size() > 0) ? 1 : 0;
			nodesOverKC[i]->inParentsListIDs.push_back(i);
			nodesOverKC[i]->parents.push_back(newNode);
			nodesOverKC[i]->accumulatedEnergy = 0.f;
			nodesOverKC[i]->computeLocalQuantities();
		}
		LOGL(a);
	}
}
