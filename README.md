# README

## Decision Tree Classifier for Iris Dataset

This code is a Python script that demonstrates the use of a Decision Tree Classifier to predict the species of iris flowers in the well-known Iris dataset. The code uses the scikit-learn library for machine learning and data analysis tasks.

### Prerequisites

Before running the code, make sure you have the following libraries installed:

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

You can install these libraries using `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Usage

1. **Data Preparation**: The script reads the Iris dataset from a CSV file named `iris1.csv`. Ensure that you have this dataset in the specified file location: `/Users/Hassaan/Desktop/Exercise 8/iris1.csv`.

2. **Running the Code**: You can run the code in a Python environment like Spyder or Jupyter Notebook. Simply execute the script, and it will perform the following tasks:

    - Load the Iris dataset.
    - Split the dataset into training and testing sets.
    - Train a Decision Tree Classifier on the training data.
    - Evaluate the model's performance on the testing data.
    - Display a confusion matrix and a heatmap.

3. **Pruning**: The code also includes a section for pruning the decision tree by varying the maximum depth parameter. This part of the code allows you to experiment with different tree depths and observe their impact on model performance.

### Code Structure

The code is divided into the following sections:

- Importing necessary libraries and loading the dataset.
- Data exploration and preprocessing.
- Splitting the data into training and testing sets.
- Training a Decision Tree Classifier with optional pruning.
- Evaluating the model's performance using accuracy and a confusion matrix.

### Results

The code provides accuracy scores and a confusion matrix to assess the performance of the Decision Tree Classifier in predicting iris species.

### Customization

Feel free to modify and customize the code to fit your specific needs. You can experiment with different hyperparameters for the Decision Tree Classifier and visualize the results in various ways.

For any questions or issues, please contact [Hassaan](mailto:hassaan@email.com).

Enjoy experimenting with the Decision Tree Classifier on the Iris dataset!
