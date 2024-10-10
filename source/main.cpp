#include "FCN.h"
#include "Network.h"
#include "ANetwork.h"
#include "MNIST.h"



int main()
{

	const int datapointS = 28 * 28;
	const int labelS = 10;

	const int testSetSize = 10000;
	float** testLabels = read_mnist_labels("MNIST\\t10k-labels-idx1-ubyte", testSetSize);
	float** testDatapoints = read_mnist_images("MNIST\\t10k-images-idx3-ubyte", testSetSize);

	const int trainSetSize = 60000;
	float** trainLabels = read_mnist_labels("MNIST\\train-labels-idx1-ubyte", trainSetSize);
	float** trainDatapoints = read_mnist_images("MNIST\\train-images-idx3-ubyte", trainSetSize);

	int batchSize = 1;
	auto [batchedPoints, batchedLabels] = create_batches(trainDatapoints, trainLabels, trainSetSize, batchSize);
	auto [testBatchedPoints, testBatchedLabels] = create_batches(testDatapoints, testLabels, testSetSize, batchSize);

	LOG(std::setprecision(4));




	bool testANetworkClass = false;
	bool testNetworkClass = true;
	bool testFCNclass = false;

	if (testANetworkClass)
	{

		ANode::xlr = .7f;

		ANode::xReg = .05f;
		ANode::wReg = .05f;

		ANode::wPriorStrength = 1.0f;

		ANode::observationImportance = 1.0f;
		ANode::certaintyDecay = 1.0f;


		int nTrainSteps = 2; 
		int nTestSteps = 4;

		constexpr bool dynamicTopology = false;

		// C++ is really stupid sometimes
		const int _nLayers = 4;
		int _sizes[_nLayers + 2] = { 0, datapointS + labelS, 20, 15, 10, 0 };
		/*const int _nLayers = 2;
		int _sizes[_nLayers + 2] = { 0, datapointS + labelS, 10, 0 };*/


		int nLayers = _nLayers;
		int* sizes = &(_sizes[1]);

		if (dynamicTopology)
		{
			nLayers = 0;
			sizes = nullptr;
		}

		ANetwork nn(datapointS, labelS, nLayers, sizes);


		bool onePerClass = true;
		onePerClass = false;
		if (onePerClass) {
			for (int u = 0; u < 1; u++) {
				for (int i = 0; i < 10; i++) {
					int id = 0;
					while (batchedLabels[id][i] != 1.0f) {
						id++;
					}
					nn.learn(batchedPoints[id], batchedLabels[id], nTrainSteps);
				}
				LOG("LOOP " << u << " done.\n\n")
			}
		}
		else {
			for (int i = 0; i < 500; i++)
			{
				nn.learn(batchedPoints[i], batchedLabels[i], nTrainSteps);
			}
		}


		int nTests = 1000;
		int nCorrects = 0;
		float* output = nn.output;
		for (int i = 0; i < nTests; i++)
		{
			nn.evaluate(testBatchedPoints[i], nTestSteps);

			LOG("\n");


			float MSE_loss = .0f;
			for (int j = 0; j < labelS; j++)
			{
				MSE_loss += powf(output[j] - testBatchedLabels[i][j], 2.0f);
			}
			int isCorrect = isCorrectAnswer(output, testBatchedLabels[i]);
			LOGL(isCorrect << " " << MSE_loss << "\n");
			nCorrects += isCorrect;
		}
		LOGL("\n" << (float)nCorrects / (float)nTests);
	}
	else if (testNetworkClass)
	{

		Node::xlr = 1.f;
		Node::wxlr = .8f;
		Node::wtlr = .3f;

		Node::wxPriorStrength = .5f;
		Node::wtPriorStrength = 1.0f;
		Node::observationImportance = .5f;
		Node::certaintyDecay = .01f;

		Node::xReg = .05f;
		Node::wxReg = .025f;
		Node::wtReg = 1.0f;
		Node::btReg = 1.0f;

		Node::lambda1 = 1.f;
		Node::lambda2 = 1.f;


	
		constexpr bool dynamicTopology = false; // TODO make the accumulatedEnergy decay of the nodes into a parameter
#ifdef DYNAMIC_PRECISIONS // good values for these parameters vary wildly if DYNAMIC_PRECISIONS is switched
		Network::KC = 4.f;
		Network::KN = 50.f;
#else
		Network::KC = 8.f;
		Network::KN = 200.f;
#endif
		// C++ is really stupid sometimes
		const int _nLayers = 4;
		int _sizes[_nLayers + 2] = {0, datapointS + labelS, 20, 15, 10, 0};
		/*const int _nLayers = 2;
		int _sizes[_nLayers + 2] = { 0, datapointS + labelS, 10, 0 };*/



		int nLayers = _nLayers;
		int* sizes = &(_sizes[1]);
		if (dynamicTopology) 
		{
			nLayers = 0;
			sizes = nullptr;
		}
		
		Network nn(datapointS, labelS, nLayers, sizes);

		int nTrainSteps = 4; // Suprisingly, less steps leads to much better results. More steps requires lower wxlr.
		int nTestSteps = 4;
		int nTests = 1000;
		
		bool onePerClass = false;
		bool onlineRandom = true;
		bool timeDependancy = false;
		if (onePerClass) {
			for (int u = 0; u < 5; u++) {
				for (int i = 0; i < 10; i++) {
					int id = 0;
					while (batchedLabels[id][i] != 1.0f) {
						id++;
					}
					nn.learn(batchedPoints[id], batchedLabels[id], nTrainSteps);
				}
				LOG("LOOP " << u << " done.\n\n")
			}
		}
		else if (onlineRandom){
			for (int i = 0; i < 500; i++)
			{
				nn.learn(batchedPoints[i], batchedLabels[i], nTrainSteps);
			}
		}
		else if (timeDependancy){
			int id = 0;
			for (int u = 0; u < 250; u++) { // 50k train datapoints so (250 * 10) < 50000 is safe in expectation
				nn.learn(batchedPoints[id], batchedLabels[id], nTrainSteps);

				int label = -1;
				for (int i = 0; i < 10; i++)
				{
					if (batchedLabels[id][i] == 1.0f) {
						label = i;
						break;
					}
				}

				int j = 1;
				while (batchedLabels[id+j][(label+1)%10] != 1.0f) {
					j++;
				}
				
				id = id + j;
				nn.learn(batchedPoints[id], batchedLabels[id], nTrainSteps);
				id++;
			}
		}

		if (timeDependancy) {
			int nCorrects = 0;
			float* output = nn.output;
			int id = 0;
			for (int u = 0; u < 500; u++) { // 10k test datapoints so (500 * 10) < 10000 is safe in expectation
				
				nn.evaluate(testBatchedPoints[id], nTestSteps);
				LOG("\n");
				float MSE_loss = .0f;
				for (int v = 0; v < labelS; v++)
				{
					MSE_loss += powf(output[v] - testBatchedLabels[id][v], 2.0f);
				}
				int isCorrect = isCorrectAnswer(output, testBatchedLabels[id]);
				LOGL(isCorrect << " " << MSE_loss << "\n");
				nCorrects += isCorrect;


				int label = -1;
				for (int i = 0; i < 10; i++)
				{
					if (batchedLabels[id][i] == 1.0f) {
						label = i;
						break;
					}
				}

				int j = 1;
				while (batchedLabels[id + j][(label + 1) % 10] != 1.0f) {
					j++;
				}

				id = id + j;
				nn.evaluate(testBatchedPoints[id], nTestSteps); 
				LOG("\n");
				MSE_loss = .0f;
				for (int v = 0; v < labelS; v++)
				{
					MSE_loss += powf(output[v] - testBatchedLabels[id][v], 2.0f);
				}
				isCorrect = isCorrectAnswer(output, testBatchedLabels[id]);
				LOGL(isCorrect << " " << MSE_loss << "\n");
				nCorrects += isCorrect;
				id++;
			}

			LOGL("\n" << (float)nCorrects / (float)nTests);
		}
		else {
			int nCorrects = 0;
			float* output = nn.output;
			for (int i = 0; i < nTests; i++)
			{
				nn.evaluate(testBatchedPoints[i], nTestSteps);

				LOG("\n");


				float MSE_loss = .0f;
				for (int j = 0; j < labelS; j++)
				{
					MSE_loss += powf(output[j] - testBatchedLabels[i][j], 2.0f);
				}
				int isCorrect = isCorrectAnswer(output, testBatchedLabels[i]);
				LOGL(isCorrect << " " << MSE_loss << "\n");
				nCorrects += isCorrect;
			}
			LOGL("\n" << (float)nCorrects / (float)nTests);
		}
		

		if (dynamicTopology)
		{
			int nAddedNodes = nn.getNNodes() - labelS - datapointS;
			LOGL("\nNetwork has added " + std::to_string(nAddedNodes) + " new nodes.\n");
		}

	} 
	else if (testFCNclass)
	{

#ifdef LABEL_IS_DATAPOINT
		const int nLayers = 4;
		int sizes[nLayers + 2] = { 0, datapointS + labelS, 20, 15, 10 , 0 };
#else
		const int nLayers = 3;
		int sizes[nLayers + 2] = { 0, datapointS, 10, labelS, 0 };
#endif


		FCN nn(nLayers, &sizes[1], datapointS);

		nn.wReg = 1.f;  // set to 1 to disable
		nn.xReg = 1.f;  // set to 1 to disable
		nn.xlr = .5f;
		nn.wlr = .2f;
		nn.certaintyDecay = .02f;


		// poor, TODO
#ifdef LABEL_IS_DATAPOINT
		float* output = &nn.x[0][datapointS];
#else
		float* output = nn.x[nLayers - 1];
#endif

		bool onePerClass = true;
		onePerClass = false;
		if (onePerClass) {
			for (int u = 0; u < 1; u++) {
				for (int i = 0; i < 10; i++) {
					int id = 0;
					while (batchedLabels[id][i] != 1.0f) {
						id++;
					}
					nn.learn(batchedPoints[id], batchedLabels[id], 10);
				}
			}
		}
		else {
			for (int i = 0; i < 1000; i++)
			{
				nn.learn(batchedPoints[i], batchedLabels[i], 5);
				if (i%100 == 0) LOGL(i)
			}
		}


		int nTests = 1000;
		int nCorrects = 0;
		for (int i = 0; i < nTests; i++)
		{
			nn.evaluate(testBatchedPoints[i], 10);
			LOG("\n");

			float MSE_loss = .0f;
			for (int j = 0; j < labelS; j++)
			{
				MSE_loss += powf(output[j] - testBatchedLabels[i][j], 2.0f);
			}
			int isCorrect = isCorrectAnswer(output, testBatchedLabels[i]);
			LOGL(isCorrect << " " << MSE_loss << "\n");
			nCorrects += isCorrect;
		}
		LOGL("\n" << (float)nCorrects / (float)nTests);
	}

}

