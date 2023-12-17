# Text Classification with Various Embedding Techniques

This repository contains a Jupyter notebook that explores text classification using different word embedding techniques. The notebook demonstrates the process of generating embeddings, training logistic regression models, and evaluating their performance.

## Contents

- `Custom_Classifier.ipynb`: The main Jupyter notebook with the entire workflow.
- `labeled.csv`: A labeled dataset generated with the help of GPT 3.5.

## Features

- Comparison of Word2Vec, BERT, OpenAI Ada 2, and other embedding techniques.
- Model training and evaluation including accuracy, precision, recall, and ROC-AUC metrics.
- Detailed visualizations for model comparison.

## Usage

To run the notebook:
1. Clone the repository.
2. Install required dependencies.
3. Run the Jupyter notebook.

You'll need to provide an OpenAI API key if you want to generate embeddings with Ada 2.

### Alternative

Go [here](https://colab.research.google.com/drive/1THEqXHIvy1IEAzVLvB0rVwwSW30jZ5Ku?usp=sharing) and run it for free on Google Colab.  You'll need to upload the `labeled.csv` data file from this repository.  A free Colab notebook can handle the model training and the BERT embeddings, no problem.

## Contributing

Contributions to improve the notebook or add more embedding techniques are welcome. Please follow the standard GitHub pull request process.

## License

[MIT License](LICENSE)

## Contact

For questions and feedback, please open an issue in the repository.
