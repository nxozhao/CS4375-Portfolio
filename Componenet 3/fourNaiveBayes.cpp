#define _USE_MATH_DEFINES
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace chrono;

// Get each value of the columns
vector<double> getColumn(int column, const vector<vector<double>> &data) {
  int size = data.size();
  vector<double> col(size); // To hold values

  for (int x = 0; x < size; x++) { // Loop through data
    col[x] = data[x][column];      // Assign values
  }

  return col; // Return
}

// For getting prior probabilities
vector<double> getPrior(const vector<vector<double>> &data, int yColumn) {
  int dataSize = data.size(); // Size
  vector<double> resultVector; // Final Vector
  vector<int> counts; // To hold category counts

  for (int x = 0; x < dataSize; x++) { // Count
    int value = (int)data[x][yColumn];
    if (counts.size() < value + 1) {
      counts.resize(value + 1, 0);
    }
    counts[value]++;
  }

  for (int i = 0; i < counts.size(); i++) { // Divide Desired Outcome(s) / The Total Number of Outcomes
    resultVector.push_back(counts[i] / (double)dataSize);
  }

  return resultVector; // Return
}

// Calculating the mean values
vector<double> getMean(const vector<vector<double>> &data, int xColumn,
                       int yColumn, int target) {
  vector<double> counts(target);
  vector<double> mean(target);

  for (int x = 0; x < data.size(); x++) {
    int columnValue = data[x][yColumn];
    mean[columnValue] += data[x][xColumn];
    counts[columnValue]++;
  }

  for (int x = 0; x < target; x++) {
    mean[x] = mean[x] / (double)counts[x];
  }

  return mean; // Return
}

// Calculating the Variance
vector<double> getVariance(const vector<vector<double>> &data, int xColumn,
                           int yColumn, int target,
                           const vector<double> &condProbs) {
  vector<double> counts(target);
  vector<double> resultVector(target);

  for (int x = 0; x < data.size(); x++) { // Summing
    int columnValue = data[x][yColumn];

    resultVector[columnValue] +=
        pow(data[x][xColumn] - condProbs[columnValue], 2);
    counts[columnValue]++;
  }

  for (int x = 0; x < target; x++) { // Averaging
    resultVector[x] = resultVector[x] / (double)counts[x];
  }

  return resultVector; // Return
}

// Calculating Likelihood
double getLikelihood(double value, double mean, double variance) {
  return 1 / sqrt(2 * M_PI * variance) *
         exp(-pow((value - mean), 2) / (2 * variance)); // Return
}

// Calculating Conditional Probabilities
vector<vector<vector<double>>>
getConditionalProbabilities(const vector<vector<double>> &data,
                            const vector<int> &xColumns, int yColumn,
                            int target) {
  int xColSize = xColumns.size();
  vector<vector<vector<double>>> resultVector(xColSize,
                                              vector<vector<double>>(target));

  for (int x = 0; x < data.size(); x++) {
    int targetColumn = data[x][yColumn];

    for (int y = 0; y < xColSize; y++) {
      double value = data[x][xColumns[y]];

      if (resultVector[y][0].size() < value + 1) {
        for (int z = 0; z < target; z++) {
          resultVector[y][z].resize(value + 1);
        }
      }

      resultVector[y][targetColumn][value]++;
    }
  }

  for (int x = 0; x < resultVector.size(); x++) {
    for (int y = 0; y < target; y++) {
      int total = 0;

      for (int z = 0; z < resultVector[x][y].size(); z++) {
        total += resultVector[x][y][z];
      }

      for (int z = 0; z < resultVector[x][y].size(); z++) {
        resultVector[x][y][z] = resultVector[x][y][z] / (double)total;
      }
    }
  }

  return resultVector; // Return
}

// Perform naive bayes
tuple<vector<double>, vector<vector<vector<double>>>,
      vector<vector<vector<double>>>>
naiveBayes(const vector<vector<double>> &data, const vector<int> &discrete,
           const vector<int> &continuous, int yColumn) {
  vector<double> priorProbs = getPrior(data, yColumn); // Get priors
  vector<vector<double>> means;
  vector<vector<double>> variances;
  vector<vector<vector<double>>> discreteLikelihoods =
      getConditionalProbabilities(data, discrete, yColumn,
                                  priorProbs.size()); // Discrete likelihoods

  for (int x = 0; x < continuous.size(); x++) { // Continuous likelihoods
    means.push_back(getMean(data, continuous[x], yColumn, priorProbs.size()));
    variances.push_back(
        getVariance(data, continuous[x], yColumn, priorProbs.size(), means[x]));
  }

  return {priorProbs, discreteLikelihoods, {means, variances}}; // Return
}

// For getting raw probabilities
vector<double>
getRawProbabilities(const vector<double> &priorProbs,
                    const vector<double> &discreteX,
                    const vector<vector<vector<double>>> &condProbs,
                    const vector<double> &continuousX,
                    const vector<vector<vector<double>>> &condMeanAndVar) {
  vector<double> resultVector;
  vector<vector<double>> condMean = condMeanAndVar[0];
  vector<vector<double>> condVar = condMeanAndVar[1];
  double denominator = 0;

  for (int x = 0; x < priorProbs.size(); x++) {
    double start = 1;

    for (int y = 0; y < condProbs.size(); y++) {
      start *= condProbs[y][x][discreteX[y]];
    }

    for (int y = 0; y < condMean.size(); y++) {
      start *= getLikelihood(continuousX[y], condMean[y][x], condVar[y][x]);
    }

    start *= priorProbs[x];
    denominator += start;
    resultVector.push_back(start);
  }

  for (int x = 0; x < resultVector.size(); x++) {
    resultVector[x] = resultVector[x] / denominator;
  }

  return resultVector; // Return
}

// For getting predictions
vector<vector<double>>
getPredictions(const tuple<vector<double>, vector<vector<vector<double>>>,
                           vector<vector<vector<double>>>> &naiveBayes,
               const vector<vector<double>> &data, const vector<int> &discrete,
               const vector<int> &continuous, int yColumn) {
  vector<vector<double>> resultVector;
  vector<double> priorProbs = get<0>(naiveBayes);
  vector<vector<vector<double>>> discreteLikelihoods = get<1>(naiveBayes);
  vector<vector<vector<double>>> continuousLikelihoods = get<2>(naiveBayes);

  for (int x = 0; x < data.size(); x++) {
    vector<double> dsicreteVals;
    vector<double> continuousVals;

    for (int y = 0; y < discrete.size(); y++) {
      dsicreteVals.push_back(data[x][discrete[y]]);
    }

    for (int y = 0; y < continuous.size(); y++) {
      continuousVals.push_back(data[x][continuous[y]]);
    }

    resultVector.push_back(
        getRawProbabilities(priorProbs, dsicreteVals, discreteLikelihoods,
                            continuousVals, continuousLikelihoods));
  }

  return resultVector; // Return
}

// Taken from fourRegression - Calculating Confusion Matrix
vector<int> getConfusionMatrix(const vector<double> &y,
                               const vector<double> &predictY) {

  int truePos = 0;
  int falsePos = 0;
  int trueNeg = 0;
  int falseNeg = 0;

  for (int x = 0; x < y.size(); x++) { // Loop
    if (y[x] == 1 && predictY[x] >= 0.5)
      truePos++;
    else if (y[x] == 0 && predictY[x] >= 0.5)
      falsePos++;
    else if (y[x] == 0 && predictY[x] < 0.5)
      trueNeg++;
    else if (y[x] == 1 && predictY[x] < 0.5)
      falseNeg++;
  }

  return {truePos, falsePos, trueNeg, falseNeg}; // Return Confusion Matrix
}

// Taken from fourRegression - Accuracy, Sensitivity, and Specificity
// For accuracy
double getAccuracy(const vector<double> &y, const vector<double> &predictY) {
  vector<int> cMatrix = getConfusionMatrix(y, predictY);
  return (cMatrix[0] + cMatrix[2]) / (double)y.size();
}

// For sensitivity
double getSensitivity(const vector<double> &y, const vector<double> &predictY) {
  vector<int> cMatrix = getConfusionMatrix(y, predictY);
  return (cMatrix[0]) / (double)(cMatrix[0] + cMatrix[3]);
}

// For specificity
double getSpecificity(const vector<double> &y, const vector<double> &predictY) {
  vector<int> cMatrix = getConfusionMatrix(y, predictY);
  return (cMatrix[2]) / (double)(cMatrix[2] + cMatrix[1]);
}

// For printing Naive Bayes model
void getNBModel(const tuple<vector<double>, vector<vector<vector<double>>>,
                            vector<vector<vector<double>>>> &naiveBayes,
                const vector<string> &headers, const vector<int> &discreteXCols,
                const vector<int> &continuousXCols) {
  vector<double> priorProbs = get<0>(naiveBayes);
  vector<vector<vector<double>>> discreteLikelihoods = get<1>(naiveBayes);
  vector<vector<vector<double>>> concreteLikelihoods = get<2>(naiveBayes);
  int priorsSize = priorProbs.size();

  cout << "========= Summary =========" << endl;
  cout << "\tA-priori Probabilities: "; // Desired Outcome(s) / The Total Number of Outcomes
  for (int x = 0; x < priorsSize; x++) {
    cout << priorProbs[x] << " ";
  }

  cout << endl;

  cout << "\n========= Conditional Probabiliites =========" << endl;

  for (int x = 0; x < discreteLikelihoods.size(); x++) {
    cout << "\t Predictor: " << headers[discreteXCols[x]] << endl; // Predictor Name

    for (int y = 0; y < priorsSize; y++) {
      cout << "\t" << y << ": ";

      for (int z = 0; z < discreteLikelihoods[x][y].size(); z++) {
        cout << discreteLikelihoods[x][y][z] << " ";
      }

      cout << endl;
    }

    cout << endl;
  }

  // Mean and Variance for "age" (the only continuous attribute)
  for (int x = 0; x < concreteLikelihoods[0].size(); x++) {
    cout << "\t Predictor: " << headers[continuousXCols[x]] << endl; // Predictor Name
    for (int y = 0; y < priorProbs.size(); y++) {
      cout << "\t Mean & Variance " << y << ": ";
      cout << concreteLikelihoods[0][x][y] << " " << concreteLikelihoods[1][x][y] << endl;
      }
  }
}

// Taken from fourRegression - For printing metrics
void printMetrics(const vector<double> &y, const vector<double> &y_pred) {
  cout << "\tAccuracy = " << getAccuracy(y, y_pred) << endl; // Accuracy
  cout << "\tSensitivity = " << getSensitivity(y, y_pred) << endl; // Sensitivity
  cout << "\tSpecificity = " << getSpecificity(y, y_pred) << endl; // Specificity
}

int main(int argc, char **argv) {
  ifstream inFS;
  string line;
  vector<vector<double>> train;
  vector<vector<double>> test;

  const string data_file = "titanic_project.csv";

  cout << "Opening file: titanic_project.csv" << endl;

  inFS.open(data_file);
  if (!inFS.is_open()) {
    cout << "Could not open file: titanic_project.csv " << endl;
    return 1;
  }

    // Can now use inFS stream like cin stream
  getline(inFS, line);

  // Echo Heading
  cout << "Heading: " << line << endl << endl;

  stringstream line_stream(line);
  vector<string> headers;
  string col;

  getline(line_stream, col, ','); // Omit ID

  while (line_stream.good()) {
    getline(line_stream, col, ',');
    headers.push_back(col);
  }

  int observations = 0;
  while (inFS.good()) {
    getline(inFS, line);
    line_stream = stringstream(line);
    vector<double> row;

    getline(line_stream, col, ','); // Omit Index

    for (int i = 0; line_stream.good(); i++) {    // Reading in the data
      getline(line_stream, col, ',');
      int val = stoi(col);

      if (i == 0) {
        val--;
      }

      row.push_back(val);
    }

     // Split to 80/20 Train/Test
    if (observations < 800) {
      train.push_back(row);
    } else {
      test.push_back(row);
    }

    observations++;
  }

  vector<int> getDiscreteX = {0, 2};
  vector<int> getConcreteX = {3};
  time_point<system_clock> start = system_clock::now(); // Set Timer to calculate Run Time
  auto model = naiveBayes(train, getDiscreteX, getConcreteX, 1);
  time_point<system_clock> end = system_clock::now();
  duration<double> totalTime = end - start; // End - Start to calculate total time

  // --------------- 1(E) --------------- //

  // Naive Bayes Model
  getNBModel(model, headers, getDiscreteX, getConcreteX);
  cout << endl;

  // Test
  vector<vector<double>> predictions =
      getPredictions(model, test, getDiscreteX, getConcreteX, 1);
  vector<double> getPredictSurvived = getColumn(1, predictions);

  // Metrics
  cout << "\n ========= Metrics =========" << endl; // hehe funny Metrics Header
  printMetrics(getColumn(1, test), getPredictSurvived);
  cout << endl;

  // Total Time
  cout << "\n ========= Run Time ========="<< endl; // hehe funny Run Time Header
  cout << "\tTime: " << totalTime.count() << " seconds" << endl;

  return 0;
}
