#include "ANetwork.h"	


ANetwork::ANetwork(int _datapointSize, int _labelSize, int nLayers, int* sizes) :
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
			// Works for i == 0 because sizes[] is padded by zeros (see main.cpp)
			// Could throw an error if adress sanitizer were enabled. But so much more convenient.
			
			// create the node at layer i
			ANode** children = nodes.data() + offset - sizes[i - 1];
			for (int j = 0; j < sizes[i]; j++) {
				nodes[offset + j] = new ANode(sizes[i - 1], children);
			}

			// give their parent's pointers to the nodes at layer (i-1)
			ANode** parents = nodes.data() + offset;
			int* inParentsListIDs = new int[sizes[i]];
			for (int j = 0; j < sizes[i-1]; j++) {
				std::fill(inParentsListIDs, inParentsListIDs + sizes[i], j);
				children[j]->registerInitialParents(parents, inParentsListIDs, sizes[i]);
			}

			offset += sizes[i];

			delete[] inParentsListIDs;
		}
	}
	else 
	{
		dynamicTopology = true;

		nodes.resize(datapointSize + labelSize);
		for (int i = 0; i < nodes.size(); i++)
		{
			nodes[i] = new ANode(0, nullptr);
		}
	}


	// Set the nodes quatities to their correct values to prepare inference
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


ANetwork::~ANetwork() 
{
	delete[] output;
	for (int i = 0; i < nodes.size(); i++)
	{
		delete nodes[i];
	}
}


// Better results when enabled. When true, asynchronous updates are performed with a random permutation.
const bool randomOrder = true;


void ANetwork::learn(float* _datapoint, float* _label, int nSteps)
{
	setActivities(_datapoint, _label);

	int nClamped = datapointSize + labelSize;
	std::vector<int> permutation(nodes.size() - nClamped);
	for (int i = 0; i < nodes.size() - nClamped; i++) permutation[i] = i + nClamped;

	float previousEnergy = computeTotalActivationEnergy();
	for (int s = 0; s < nSteps; s++) 
	{
		LOG(previousEnergy);

		if (randomOrder) {
			
			std::shuffle(permutation.begin(), permutation.end(), generator);
			for (int i = 0; i < nodes.size() - nClamped; i++)
			{
				nodes[permutation[i]]->updateActivation();
			}
		} else 
		{
			for (int i = nClamped; i < nodes.size(); i++)
			{
				nodes[i]->updateActivation();
			}
		}


		float currentEnergy = computeTotalActivationEnergy();
		previousEnergy = currentEnergy;
	}
	LOG(previousEnergy << "\n\n");

	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->updateIncomingWeights();
		nodes[i]->calcifyIncomingWeights();
	}
}

void ANetwork::evaluate(float* _datapoint, int nSteps)
{
	setActivities(_datapoint, nullptr);

	int nClamped = datapointSize;
	std::vector<int> permutation(nodes.size() - nClamped);
	for (int i = 0; i < nodes.size() - nClamped; i++) permutation[i] = i + nClamped;

	float previousEnergy = computeTotalActivationEnergy();
	for (int s = 0; s < nSteps; s++)
	{
		LOG(previousEnergy);

		if (randomOrder) {

			std::shuffle(permutation.begin(), permutation.end(), generator);
			for (int i = 0; i < nodes.size() - nClamped; i++)
			{
				nodes[permutation[i]]->updateActivation();
			}
		}
		else
		{
			for (int i = nClamped; i < nodes.size(); i++)
			{
				nodes[i]->updateActivation();
			}
		}


		float currentEnergy = computeTotalActivationEnergy();
		previousEnergy = currentEnergy;
	}
	LOG(previousEnergy << "\n\n");
}


float ANetwork::computeTotalActivationEnergy()
{
	float E2 = 0.f;

	for (int i = 0; i < nodes.size(); i++)
	{
		E2 += nodes[i]->epsilon * nodes[i]->epsilon;
		// + constants
	}

	return E2 *.5f;
}


void ANetwork::setActivities(float* _datapoint, float* _label)
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
}


