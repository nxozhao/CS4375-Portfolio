#define _USE_MATH_DEFINES
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace chrono;

const double learnRate = 0.001;
const double epsilon = 0.001;

/* ========== 1(B) ========== */

// Get each value of the columns
vector<double> getColumn(int column, const vector<vector<double>> &data) {
  int size = data.size();
  vector<double> col(size); // To hold values

  for (int x = 0; x < size; x++) { // Loop through data
    col[x] = data[x][column];      // Assign values
  }

  return col; // Return
}

// For transposting the data matrix in order to calculate matric multiplication
vector<vector<double>> getTranspose(const vector<vector<double>> &matrix) {
  int sizeOne = matrix[0].size();
  int sizeTwo = matrix.size();
  vector<vector<double>> transpose(
      sizeOne, vector<double>(sizeTwo)); // To hold transpose

  for (int x = 0; x < sizeTwo; x++) { // Loop through data
    for (int y = 0; y < sizeOne; y++) {
      transpose[y][x] = matrix[x][y]; // Assign values
    }
  }

  return transpose; // output transpose
}

/* Matrix Multiplicaton */

// For product of two dimensional matrix and vector
vector<double> getMultiply(const vector<vector<double>> &matrix,
                           const vector<double> &vec) {
  int sizeOne = matrix.size();
  int sizeTwo = vec.size();
  vector<double> product(sizeOne); // To hold the product

  for (int x = 0; x < sizeOne; x++) { // Loop through given matrices
    double sum = 0;                   // To hold sum
    for (int y = 0; y < sizeTwo; y++) {
      sum += matrix[x][y] * vec[y]; // Arithmetic
    }
    product[x] = sum; // Assign
  }

  return product; // output product
}

// For dot product of two vectors
double getMultiply(const vector<double> &vectorOne,
                   const vector<double> &vectorTwo) {
  double dotProduct = 0; // To hold the product

  for (int x = 0; x < vectorOne.size(); x++) { // Loop through values
    dotProduct += vectorOne[x] * vectorTwo[x]; // Arithmetic
  }

  return dotProduct; // output dotProduct
}

// For product of two dimensional matrix and scalar
vector<vector<double>> getMultiply(const vector<vector<double>> &matrix,
                                   double scalar) {
  int mXSize = matrix.size();
  int mYSize = matrix[0].size();
  vector<vector<double>> product(mXSize,
                                 vector<double>(mYSize)); // To hold the product

  for (int x = 0; x < mXSize; x++) { // Loop through values
    for (int y = 0; y < mYSize; y++) {
      product[x][y] = matrix[x][y] * scalar; // Arithmetic
    }
  }

  return product; // output product
}

// For product of vector and scalar (vec * scalar)
vector<double> getMultiply(const vector<double> &vec, double scalar) {
  int vectorSize = vec.size();
  vector<double> product(vectorSize); // To hold the product

  for (int x = 0; x < vectorSize; x++) { // Loop through values
    product[x] = vec[x] * scalar;        // Arithmetic
  }

  return product; // output product
}

// For vector subtraction (vectorOne - vectorTwo)
vector<double> getSubtraction(const vector<double> &vectorOne,
                              const vector<double> &vectorTwo) {
  int vecOneSize = vectorOne.size();
  vector<double> result(vecOneSize); // To hold the result

  for (int x = 0; x < vecOneSize; x++) {     // Loop through values
    result[x] = vectorOne[x] - vectorTwo[x]; // Arithmetic
  }

  return result; // output result
}

// For log odds with weights
vector<double> sigmoid(const vector<vector<double>> &matrix,
                       const vector<double> &test) {
  int matrixSize = matrix.size();
  vector<double> resultVector(matrixSize); // To hold ther resulting vector

  for (int x = 0; x < matrixSize; x++) { // Loop through values
    resultVector[x] = 1 / (1 + pow(M_E, -getMultiply(matrix[x], test)));
  }

  return resultVector; // output final vector
}

// Logistic Regression - Train
vector<double> gradientDescent(const vector<vector<double>> &matrix,
                               const vector<double> &vec) {
  vector<double> testValue(matrix[0].size()); // To hold the result
  vector<vector<double>> matrixTranspose =
      getTranspose(matrix); // Get transpose

  while (true) { // Loop until break
    vector<double> previousTestValue = testValue;
    testValue = getSubtraction(
        testValue,
        getMultiply(
            getMultiply(matrixTranspose,
                        getSubtraction(sigmoid(matrix, testValue), vec)),
            learnRate)); // Get new weights
    bool needBreak = true;

    for (int x = 0; x < testValue.size(); x++) {
      if (abs(testValue[x] - previousTestValue[x]) > epsilon) {
        needBreak = false; // to continue training
        break;
      }
    }

    if (needBreak) { // Exit condition
      break;
    }
  }

  return testValue; // Return
}

/* ========== 1(B) ========== */

vector<double> logisticRegression(const vector<vector<double>> &data,
                                  const vector<int> &xColumn, int yColumn) {
  int dataSize = data.size();
  int xColoumnSize = xColumn.size();
  vector<vector<double>> xMatrix(
      dataSize, vector<double>(xColoumnSize + 1, 1)); // To hold x
  vector<double> yVector(dataSize);                   // To hold y

  for (int x = 0; x < dataSize; x++) {       // Loop through values
    for (int y = 0; y < xColoumnSize; y++) { // Get x
      xMatrix[x][y + 1] = data[x][xColumn[y]];
    }

    yVector[x] = data[x][yColumn]; // Get y
  }

  return gradientDescent(xMatrix, yVector); // Perform gradient descent
}

// For predictions vector
vector<double> getPrediction(const vector<vector<double>> &data,
                             const vector<int> &xColumn,
                             const vector<double> &testValue) {
  int dataSize = data.size();
  int xColSize = xColumn.size();
  vector<vector<double>> matrix(
      dataSize, vector<double>(xColSize + 1, 1)); // To hold vector

  for (int x = 0; x < dataSize; x++) { // Loop through values
    for (int y = 0; y < xColSize; y++) {
      matrix[x][y + 1] = data[x][xColumn[y]];
    }
  }

  // Begin Log Likelihoods
  vector<double> logLikelihood = getMultiply(matrix, testValue);
  vector<double> result(logLikelihood.size());

  for (int x = 0; x < logLikelihood.size(); x++) { // Convert to probabilities
    result[x] = exp(logLikelihood[x]) / (1 + exp(logLikelihood[x]));
  }

  return result; // Return
}

// Calculating Confusion Matrix
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

/* ========== 1(D) ========== */

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

// For printing metrics
void printMetrics(const vector<double> &y, const vector<double> &predictY) {
  cout << "\tAccuracy = " << getAccuracy(y, predictY) << endl;
  cout << "\tSensitivity = " << getSensitivity(y, predictY) << endl;
  cout << "\tSpecificity = " << getSpecificity(y, predictY) << endl;
}

// For printing coefficients
void printCoefficients(const vector<string> &headers,
                       const vector<int> &xColumn,
                       const vector<double> &testValue) {
  cout << "\tIntercept = " << testValue[0] << endl;

  for (int x = 1; x < testValue.size(); x++) {
    cout << "\t" << headers[xColumn[x - 1]] << " = " << testValue[x] << endl;
  }
}

int main(int argc, char **argv) {
  ifstream inFS;
  string line;
  vector<vector<double>> train;
  vector<vector<double>> test;

  /* ========== 1(A) ========== */
  cout << "Opening file: titanic_project.csv." << endl;

  inFS.open("titanic_project.csv");
  if (!inFS.is_open()) {
    cout << "Could not open file: titanic_project.csv." << endl;
    return 1; // 1 indicates error
  }

  // Can now use inFS stream like cin stream
  getline(inFS, line);

  // Echo Heading
  cout << "Heading: " << line << endl;

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

    getline(line_stream, col, ','); // Omit index

    while (line_stream.good()) {
      getline(line_stream, col, ',');
      row.push_back(stoi(col));
    }

    // Split to 80/20 Train/Test
    if (observations < 800) {
      train.push_back(row);
    } else {
      test.push_back(row);
    }

    observations++;
  }

  vector<int> xColumn = {2};
  time_point<system_clock> start =
      system_clock::now(); // Set Timer to calculate Run Time
  vector<double> testValue = logisticRegression(train, xColumn, 1);
  time_point<system_clock> end = system_clock::now();
  duration<double> totalTime = end - start; // End - Start to calculate total time

  // ========== 1(E) ========== //

  // Coefficients
  cout << "\n ========= Coefficients =========" << endl; // hehe funny Coefficients Header
  printCoefficients(headers, xColumn, testValue);
  cout << endl;

  // Test
  vector<double> predictions = getPrediction(test, xColumn, testValue);

  // Metrics
  cout << "\n ========= Metrics =========" << endl; // hehe funny Metrics Header
  printMetrics(getColumn(1, test), predictions);
  cout << endl;

  // Total Time
cout << "\n ========= Run Time =========" << endl;  // hehe funny Run Time Header
  cout << "\tTime: " << totalTime.count() << " seconds" << endl;

  return 0;
}
