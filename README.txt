The purpose of this file is providing an explained index of the content of this repository.

For the specifics of files with really close names, please refer to https://github.com/tetuante/solarNodeCasting/tree/master/merger

Configuration files are located under the "jsons" folder, matching the name of the file with its corresponding .py file
Regarding the naming of PCA configuration files, the number of clusters is stated in letters, then the number of PCA dimensions
and finally the number of the specific cluster.
For every configuration file appended with "_base", there is an equivalent without it because the ones with are used for testing
the original model with each specific scenario.

nts_b.py
    Same as nts.py from the original project.

nts_filter_xcor_mor.py
    nts.py modified for the morning model, checking the data belong to the morning time frame.

nts_filter_xcor.py
    nts.py modified for the winter model, checking dates belong to this season.

xcorr.py
    Computes the cross correlation between all stations, creating histograms and some metrics for this histograms.

NN_matrices_b.py
    Added to the original code to ensure the test days are the same for the original model and the morning model.

NN_matrices_winter.py
    Ensures the same dates for the test data set in both the winter and original models.

NN_matrices_winter_base.py
   Ensures the data to test from the winter model with the original model is the same as the one used with the winter model.

pca.py
    From the reference data set, applies PCA and k-means clustering in order to create the subsets of data
    as specified by its configuration file, similar to NN_matrices.py.

NN_models_keras_se18.py
    Same function as NN_models.py from the original project. Trains the model, here using Keras instead of scikit-learn
    Used for the original model and the ones resulting from cross correlation.

NN_models_keras_se18_c.py
    Same as NN_prediction_graphs_keras_se18.py for the models defined by PCA and clustering.

NN_prediction_graphs_keras_se18.py
    Similar to NN_models_keras_se18.py, for predicting instead of training.

NN_prediction_graphs_keras_se18_cY.py
    Similar to NN_models_keras_se18_c.py, for predicting instead of training.


    
