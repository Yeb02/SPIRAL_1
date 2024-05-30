#include "FCN.h"
#include "Network.h"
#include "MNIST.h"


int isCorrectAnswer(float* out, float* y) {
	float outMax = -1000.0f;
	int imax = -1;
	for (int i = 0; i < 10; i++)
	{
		if (outMax < out[i]) {
			outMax = out[i];
			imax = i;
		}
	}

	return y[imax] == 1.0f ? 1 : 0;
}

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

	LOG(std::setprecision(3));

	bool dynamicTopology = true;

	if (dynamicTopology)
	{
		Node::priorStrength = .3f;				 // > 0
		Node::activationDescentStepSize = .3f;	 // > 0
		Node::observationImportance = 1.0f;      // > 0
		Node::certaintyDecay = .95f;			 // <= 1
		Node::weightRegularization = 1.0f;		 // <= 1

		Network nn(datapointS, labelS);

		
		bool onePerClass = true;
		onePerClass = false;
		if (onePerClass) {
			for (int u = 0; u < 1; u++) {
				for (int i = 0; i < 10; i++) {
					int id = 0;
					while (batchedLabels[id][i] != 1.0f) {
						id++;
					}
					nn.asynchronousLearn(batchedPoints[id], batchedLabels[id], 10);
				}
			}
		}
		else {
			for (int i = 0; i < 1000; i++)
			{
				nn.asynchronousLearn(batchedPoints[i], batchedLabels[i], 5);
			}
		}


		int nTests = 1000;
		int nCorrects = 0;
		float* output = nn.output;
		for (int i = 0; i < nTests; i++)
		{
			nn.asynchronousEvaluate(testBatchedPoints[i], 10);
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

		float weightRegularization = 1.0f; // set to 1 to disable
		float certaintyDecay = .02f;
		float gradientStepSize = .5f;

		//const int _nLayers, int* _sizes, int _datapointSize, float _weightRegularization, float _gradientStepSize, float _certaintyDecay
		FCN nn(nLayers, &sizes[1], datapointS, weightRegularization, gradientStepSize, certaintyDecay);


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

