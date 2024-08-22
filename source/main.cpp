#include "FCN.h"
#include "Network.h"
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





	bool testNetworkClass = true;
	
	if (testNetworkClass)
	{

		Node::xlr = .8f;
		Node::wxlr = .3f;
		Node::wtlr = .1f;

		Node::wxPriorStrength = .2f;
		Node::wtPriorStrength = 3.f;
		Node::observationImportance = 1.f;
		Node::certaintyDecay = .98f;

		Node::xReg = .1f;
		Node::wxReg = .0f;
		Node::wtReg = .0f;

		int nTrainSteps = 3; // Suprisingly, less steps leads to much better results. More step require lower wxlr.
		int nTestSteps = 5;
		
		constexpr bool dynamicTopology = false;
		constexpr bool synchronizedDescent = false;
		
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

		
		bool onePerClass = true;
		onePerClass = false;
		if (onePerClass) {
			for (int u = 0; u < 3; u++) {
				for (int i = 0; i < 10; i++) {
					int id = 0;
					while (batchedLabels[id][i] != 1.0f) {
						id++;
					}
					nn.asynchronousLearn(batchedPoints[id], batchedLabels[id], 10);
				}
				LOG("LOOP " << u << " done.\n\n")
			}
		}
		else {
			for (int i = 0; i < 500; i++)
			{
				if (synchronizedDescent) nn.synchronousLearn(batchedPoints[i], batchedLabels[i], nTrainSteps);
				else nn.asynchronousLearn(batchedPoints[i], batchedLabels[i], nTrainSteps);
			}
		}


		int nTests = 1000;
		int nCorrects = 0;
		float* output = nn.output;
		for (int i = 0; i < nTests; i++)
		{
			if (synchronizedDescent) nn.synchronousEvaluate(testBatchedPoints[i], nTestSteps);
			else nn.asynchronousEvaluate(testBatchedPoints[i], nTestSteps);
			
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
	else {

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

