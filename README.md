# Alien Pet Health Classifier

I used this CSI4106 artificial intelligence assignment to study how neural-network capacity and regularization affect classification on tabular data. The task is to predict the health outcome in a fictional alien-pet dataset and compare models using held-out data.

The notebook follows the experiment from data preparation through model selection and evaluation.

## What it explores

- Validate the dataset's expected fields and separate features from the target.
- Split the data into training, validation, and test sets before preprocessing.
- Scale numerical features and encode the habitat category.
- Compare a small feed-forward network with a larger network that can overfit.
- Investigate early stopping, L2 regularization, and dropout.
- Compare learning curves, precision, recall, F1, and ROC-AUC.
- Select a model using validation results and inspect its test-set performance.

## Project context

This is an academic modelling experiment using TensorFlow/Keras and scikit-learn. My focus was understanding generalization and explaining model behaviour through the recorded experiments.

The [notebook](src/alien-pet-health-deep-learning.ipynb) includes the analysis and plots. The dataset and assignment context come from the [CSI4106 course materials](https://github.com/turcotte/csi4106-f25).

## License

See [LICENSE](LICENSE).
