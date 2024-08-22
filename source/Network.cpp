#include "Network.h"	


Network::Network(int _datapointSize, int _labelSize, int _nLayers, int* _sizes) :
	datapointSize(_datapointSize), labelSize(_labelSize), nLayers(_nLayers), sizes(_sizes)
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
			// Will throw an error if adress sanitizer is on though. But so much more convenient.
			Node** children = new Node * [sizes[i - 1]]; 
			std::copy(nodes.data() + offset - sizes[i - 1], nodes.data() + offset, children);

			for (int j = 0; j < sizes[i]; j++) {
				nodes[offset + j] = new Node(sizes[i - 1], children);
			}

			offset += sizes[i];
			delete[] children;
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


Network::~Network() 
{
	delete[] output;
	for (int i = 0; i < nodes.size(); i++)
	{
		delete nodes[i];
	}
}


// Better when enabled. When true, asynchronous updates are performed with a random permutation.
const bool randomOrder = true;

void Network::asynchronousLearn(float* _datapoint, float* _label, int nSteps)
{
	setActivities(_datapoint, _label);

	int nClamped = datapointSize + labelSize;

	float previousEnergy = computeTotalActivationEnergy();
	for (int s = 0; s < nSteps; s++) 
	{
		LOG(previousEnergy);

		if (randomOrder) {
			std::vector<int> permutation1(nClamped);
			for (int i = 0; i < nClamped; i++) permutation1[i] = i;
			std::shuffle(permutation1.begin(), permutation1.end(), generator);
			for (int i = 0; i < nClamped; i++)
			{
				nodes[permutation1[i]]->asynchronousGradientStep_WB_only();
			}

			std::vector<int> permutation2(nodes.size() - nClamped);
			for (int i = 0; i < nodes.size() - nClamped; i++) permutation2[i] = i + nClamped;
			std::shuffle(permutation2.begin(), permutation2.end(), generator);
			for (int i = 0; i < nodes.size() - nClamped; i++)
			{
				nodes[permutation2[i]]->asynchronousGradientStep();
			}
		} else 
		{
			for (int i = 0; i < nClamped; i++)
			{
				nodes[i]->asynchronousGradientStep_WB_only();
			}
			for (int i = nClamped; i < nodes.size(); i++)
			{
				nodes[i]->asynchronousGradientStep();
			}
		}


		float currentEnergy = computeTotalActivationEnergy();
		previousEnergy = currentEnergy;
	}
	LOG(previousEnergy << "\n\n");

	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->calcifyWB();
	}
}

void Network::asynchronousEvaluate(float* _datapoint, int nSteps) 
{
	setActivities(_datapoint, nullptr);

	int nClamped = datapointSize;

	float previousEnergy = computeTotalActivationEnergy();
	for (int s = 0; s < nSteps; s++)
	{
		LOG(previousEnergy);

		if (randomOrder) {

			std::vector<int> permutation2(nodes.size() - nClamped);
			for (int i = 0; i < nodes.size() - nClamped; i++) permutation2[i] = i + nClamped;
			std::shuffle(permutation2.begin(), permutation2.end(), generator);
			for (int i = 0; i < nodes.size() - nClamped; i++)
			{
				nodes[permutation2[i]]->asynchronousGradientStep_X_only();
			}
		}
		else
		{
			for (int i = nClamped; i < nodes.size(); i++)
			{
				nodes[i]->asynchronousGradientStep_X_only();
			}
		}

		float currentEnergy = computeTotalActivationEnergy();
		previousEnergy = currentEnergy;
	}

	LOG(previousEnergy);

	for (int i = 0; i < labelSize; i++)
	{
		output[i] = nodes[datapointSize + i]->x;
	}

}



void Network::synchronousLearn(float* _datapoint, float* _label, int nSteps)
{
	setActivities(_datapoint, _label);

	int nClamped = datapointSize + labelSize;

	float previousEnergy = computeTotalActivationEnergy();
	for (int s = 0; s < nSteps; s++)
	{
		LOG(previousEnergy);

		if (randomOrder) {
			std::vector<int> permutation1(nClamped);
			for (int i = 0; i < nClamped; i++) permutation1[i] = i;
			std::shuffle(permutation1.begin(), permutation1.end(), generator);
			for (int i = 0; i < nClamped; i++)
			{
				nodes[permutation1[i]]->synchronousGradientStep_WB_only();
			}

			std::vector<int> permutation2(nodes.size() - nClamped);
			for (int i = 0; i < nodes.size() - nClamped; i++) permutation2[i] = i + nClamped;
			std::shuffle(permutation2.begin(), permutation2.end(), generator);
			for (int i = 0; i < nodes.size() - nClamped; i++)
			{
				nodes[permutation2[i]]->synchronousGradientStep();
			}
		}
		else
		{
			for (int i = 0; i < nClamped; i++)
			{
				nodes[i]->synchronousGradientStep_WB_only();
			}
			for (int i = nClamped; i < nodes.size(); i++)
			{
				nodes[i]->synchronousGradientStep();
			}
		}


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

		float currentEnergy = computeTotalActivationEnergy();
		previousEnergy = currentEnergy;
	}
	LOG(previousEnergy << "\n\n");

	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->calcifyWB();
	}
}

void Network::synchronousEvaluate(float* _datapoint, int nSteps)
{
	setActivities(_datapoint, nullptr);

	int nClamped = datapointSize;

	float previousEnergy = computeTotalActivationEnergy();
	for (int s = 0; s < nSteps; s++)
	{
		LOG(previousEnergy);

		if (randomOrder) {

			std::vector<int> permutation2(nodes.size() - nClamped);
			for (int i = 0; i < nodes.size() - nClamped; i++) permutation2[i] = i + nClamped;
			std::shuffle(permutation2.begin(), permutation2.end(), generator);
			for (int i = 0; i < nodes.size() - nClamped; i++)
			{
				nodes[permutation2[i]]->synchronousGradientStep_X_only();
			}
		}
		else
		{
			for (int i = nClamped; i < nodes.size(); i++)
			{
				nodes[i]->synchronousGradientStep_X_only();
			}
		}

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
		nodes[i]->computeLocalQuantities(); // just in case ?

		E2 += .5f * nodes[i]->epsilon * nodes[i]->epsilon * nodes[i]->tau;
		Elog += - logf(nodes[i]->tau); 
		// + constants
	}

	return E2 *.5f + Elog;
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


