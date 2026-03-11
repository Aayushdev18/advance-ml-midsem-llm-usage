# Toy Dataset README

## Toy Dataset: Two Moons (make_moons)

This dataset was generated using `sklearn.datasets.make_moons`. It produces two interleaving half circles, providing a standard toy non-linear binary classification problem.

### How it is obtained:
The dataset is generated algorithmically and saved locally as `dataset.csv` using a script `generate_dataset.py` with a fixed random seed (42) for exact reproducibility. It contains 2,000 samples and 2 feature variables.

### How it is used:
- In `task_2_1.ipynb`, the dataset is loaded from this CSV and its properties are justified for evaluating Random Fourier features.
- In `task_2_2.ipynb`, it is used to train and test the implemented `RandomFourierFeatures` model alongside a linear classifier.
- In `task_2_3.ipynb` and onwards, it is used to visualize performance accuracy gaps vs. explicit kernel SVMs, and provides the base evaluation for ablation models.
