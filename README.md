# Learning-Wasserstein-Embeddings
Keras implementation of Deep Wasserstein Embeddings ([Learning Wasserstein Embeddings](https://arxiv.org/abs/1710.07457)) which approximates the Wasserstein distance with an autoencoder.

## Prerequisite
- Python 2.7 
- Scipy, numpy
- [Keras](https://keras.io/)
- [POT](http://pot.readthedocs.io/en/stable/)
- pylab

## Usage

First, compute the Wasserstein pairwise distance on n_pairwise couples of samples of the dataset of your choice. Previous records for mnist and quick draw are available in [data][./data]

	 $ python run_emd.py --dataset_name mnist --n_pairwise 10000

To train a siamese network to predict the pairwise wasserstein distance, use build_model.py : 

	 $ python build_model.py --dataset_name mnist --embedding_size 50 --batch_size 32 --epochs 100

## Examples
You can use your autoencoder to compute PCA, Barycenter Estimation or Interpolation with Wasserstein. The file test_model.py allows you to test your previously trained model to test on of these features or
the MSE of your network, using the method_name attributes [MSE, PCA, BARYCENTER, INTERPOLATION].

	$ python test_model.py --dataset_name mnist --method_name MSE


