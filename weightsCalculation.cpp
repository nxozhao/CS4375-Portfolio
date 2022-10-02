#include <bits/stdc++.h>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

double sigmoid(double z) { return 1.0 / (1 + exp(-z)); }

int main(int argc, char **argv) {

  ifstream inFS; // Input file stream
  string line;
  string rm_in, medv_in;
  const int MAX_LEN = 1046;

  // Try to open file
  cout << "Opening file titanic_project.csv." << endl;

  inFS.open("titanic_project.csv");
  if (!inFS.is_open()) {
    cout << "Could not open file titanic_project.csv." << endl;
    return 1; // 1 indicates error
  }

  cout << "Reading Line 1" << endl;
  getline(inFS, line);

  // Start Project
  double data_matrix[MAX_LEN][2];
  int labels[MAX_LEN];

  for (int i = 0; i < MAX_LEN; i++) {
    data_matrix[i][0] = 1;
    int j = 0;
    string line, token;
    getline(inFS, line);

    // Line to File
    stringstream str(line);
    while (getline(str, token, ',')) {
      if (j == 2) {

        labels[i] = stoi(token);
      }
      if (j == 3) {
        data_matrix[i][1] = stoi(token);
      }
      j++;
    }
  }

  double weights[2] = {1, 1};
  double learning_rate = 0.001;

  for (int i = 1; i <= 500000; i++) {
    // probability vector
    double prob_vector[MAX_LEN];

    // matrix multiplication of data_matrix and weights
    for (int j = 0; j < MAX_LEN; j++) {
      double element =
          data_matrix[j][0] * weights[0] + data_matrix[j][1] * weights[1];
      prob_vector[j] = sigmoid(element);
    }

    // error vector
    double error[MAX_LEN];
    for (int j = 0; j < MAX_LEN; j++) {
      error[j] = labels[j] - prob_vector[j];
    }

    // gradient descent
    for (int j = 0; j < 2; j++) {
      double weighted_error = 0;
      for (int k = 0; k < MAX_LEN; k++) {
        weighted_error += data_matrix[k][j] * error[k];
      }
      weights[j] += learning_rate * weighted_error;
    }
  }

  cout << "Weights = " << weights[0] << " " << weights[1] << endl;

  return 0;
}