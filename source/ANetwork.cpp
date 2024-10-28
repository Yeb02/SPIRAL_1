#include "ANetwork.h"	



ANetwork::ANetwork(int _datapointSize, int _labelSize) :
	datapointSize(_datapointSize), labelSize(_labelSize)
{
	output = new float[labelSize];

	assemblies.resize(2);

	assemblies[0] = new Assembly(datapointSize, .0f, .0f, .0f, .0f);
	assemblies[0]->firstNodeID = 0;
	assemblies[1] = new Assembly(labelSize, .0f, .0f, .0f, .0f);
	assemblies[1]->firstNodeID = datapointSize;

	nodes.resize(datapointSize + labelSize);

	for (int i = 0; i < datapointSize; i++)
	{
		nodes[i] = new ANode(assemblies[0]);
		nodes[i]->localXReg = 0.f; // no regularisation for the observations.
	}

	for (int i = datapointSize; i < datapointSize + labelSize; i++)
	{
		nodes[i] = new ANode(assemblies[1]);
		nodes[i]->localXReg = 0.f; // no regularisation for the labels.
	}
}


ANetwork::~ANetwork()
{
	delete[] output;
	for (int i = 0; i < nodes.size(); i++)
	{
		delete nodes[i];
	}
	for (int i = 0; i < assemblies.size(); i++)
	{
		delete assemblies[i];
	}
}


void ANetwork::learn(float* _datapoint, float* _label, int nSteps)
{
	setActivities(_datapoint, _label);

	float previousEnergy = computeTotalActivationEnergy();

	for (int s = 0; s < nSteps; s++)
	{
		LOG(previousEnergy);

		std::shuffle(permutation.begin(), permutation.end(), generator);
		for (int i = 0; i < permutation.size(); i++)
		{
			nodes[permutation[i]]->updateActivation();
			//nodes[permutation[i]]->setTemporaryWB();
		}

		float currentEnergy = computeTotalActivationEnergy();
		previousEnergy = currentEnergy;
	}
	LOG(previousEnergy << "\n\n");


	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->setTemporaryWB();
	}
	

	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->calcifyWB();
	}
}

void ANetwork::evaluate(float* _datapoint, int nSteps)
{
	setActivities(_datapoint, nullptr);

	float previousEnergy = computeTotalActivationEnergy();

	for (int s = 0; s < nSteps; s++)
	{
		LOG(previousEnergy);

		std::shuffle(permutation.begin(), permutation.end(), generator);
		for (int i = 0; i < permutation.size(); i++)
		{
			nodes[permutation[i]]->updateActivation();
		}

		float currentEnergy = computeTotalActivationEnergy();
		previousEnergy = currentEnergy;
	}
	LOG(previousEnergy << "\n\n");


	for (int i = 0; i < labelSize; i++)
	{
		output[i] = nodes[datapointSize + i]->x;
	}
}



float ANetwork::computeTotalActivationEnergy()
{
	float E2 = 0.f;

	for (int i = 0; i < nodes.size(); i++)
	{
		E2 += powf(nodes[i]->x - nodes[i]->mu, 2.0f);
	}

	return E2;
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


void ANetwork::readyForLearning() {
	for (int i = 0; i < datapointSize + labelSize; i++) nodes[i]->isFree = false;

	int nClamped = datapointSize + labelSize;
	permutation.resize(nodes.size() - nClamped);
	for (int i = nClamped; i < nodes.size(); i++) permutation[i - nClamped] = i;

	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->prepareToReceivePredictions();
	}
	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->transmitPredictions();
	}
}

void ANetwork::readyForTesting() {
	for (int i = datapointSize; i < datapointSize + labelSize; i++) nodes[i]->isFree = true;

	int nClamped = datapointSize + labelSize; // technically, the label nodes are no clamped. But they do not need an update, so we will ignore them.
	permutation.resize(nodes.size() - nClamped);
	for (int i = nClamped; i < nodes.size(); i++) permutation[i - nClamped] = i;

	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->prepareToReceivePredictions();
	}
	for (int i = 0; i < nodes.size(); i++)
	{
		nodes[i]->transmitPredictions();
	}
}


void ANetwork::addAssembly(Assembly* assembly)
{
	assembly->firstNodeID = assemblies[assemblies.size() - 1]->firstNodeID + assemblies[assemblies.size() - 1]->nNodes;
	assemblies.push_back(assembly);

	nodes.resize(nodes.size() + assembly->nNodes);

	for (int i = (int) nodes.size() - assembly->nNodes; i < nodes.size(); i++)
	{
		nodes[i] = new ANode(assemblies[assemblies.size()-1]);
	}
}

void ANetwork::addConnexion(int originID, int destinationID, float p)
{
	Assembly* oA = assemblies[originID];
	Assembly* dA = assemblies[destinationID];


	if (p == 1) { // TODO stupid to still use inParentsIDs in this case, should be a different prepoc directive, but not all assemblies are dense...
		std::vector<ANode*> children(dA->nNodes);
		std::vector<ANode*> parents(oA->nNodes);
		std::vector<int> inParentsIDs(oA->nNodes);
		

		for (int i = 0; i < oA->nNodes; i++)
		{
			int nc = 0;
			for (int j = 0; j < dA->nNodes; j++)
			{
				if (dA == oA && i == j) [[unlikely]] {continue; } // a node can't connect to itself.

				children[nc] = nodes[dA->firstNodeID + j];
				nc++;
			}

			nodes[oA->firstNodeID + i]->addChildren(children.data(), nc);
		}

		for (int i = 0; i < dA->nNodes; i++)
		{
			int np = 0;
			for (int j = 0; j < oA->nNodes; j++)
			{
				if (dA == oA && i == j) [[unlikely]] {continue; } // a node can't connect to itself.

				parents[np] = nodes[oA->firstNodeID + j];
				inParentsIDs[np] = i;
				np++;
			}
			nodes[dA->firstNodeID + i]->addParents(parents.data(), inParentsIDs.data(), np);
		}
	}
	else {
		std::vector<ANode*> children(dA->nNodes);

		std::vector<ANode*> parents(oA->nNodes * dA->nNodes);
		std::vector<int> inParentsIDs(oA->nNodes * dA->nNodes);
		std::vector<int> nNewParents(dA->nNodes);
		std::fill(nNewParents.begin(), nNewParents.end(), 0);

		for (int i = 0; i < oA->nNodes; i++)
		{
			int nc = 0;
			for (int j = 0; j < dA->nNodes; j++)
			{
				if (UNIFORM_01 < p) {
					if (dA == oA && i == j) [[unlikely]] {continue; } // a node can't connect to itself.

					children[nc] = nodes[dA->firstNodeID + j];
					parents[j * oA->nNodes + nNewParents[j]] = nodes[oA->firstNodeID + i];
					inParentsIDs[j * oA->nNodes + nNewParents[j]] = nc;

					nNewParents[j]++;
					nc++;
				}
			}

			nodes[oA->firstNodeID + i]->addChildren(&children[0], nc);
		}

		for (int j = 0; j < dA->nNodes; j++)
		{
			nodes[dA->firstNodeID + j]->addParents(&parents[j * oA->nNodes], &inParentsIDs[j * oA->nNodes], nNewParents[j]);
		}
	}
	
}
