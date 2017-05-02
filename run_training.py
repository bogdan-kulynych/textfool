import numpy as np
import keras

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

from data_utils import load as load_data
from model import build_model

np.random.seed(42)


epochs = 16
batch_size = 64

model_path = './data/model.dat'
log_dir = './logs'
grid_search = False


if __name__ == '__main__':
    # Load Twitter gender data
    (X_train, y_train, X_test, y_test), _ = load_data('twitter_gender_data')

    # Take a look at the shapes
    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)
    print('X_test:', X_test.shape)
    print('y_test:', y_test.shape)

    if not grid_search:
        tb_callback = keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=0, write_graph=True)

        model = build_model()
        model.fit(X_train, y_train,
                  validation_data=(X_test, y_test),
                  batch_size=batch_size, epochs=epochs,
                  callbacks=[tb_callback])

        print('Saving model weights...')
        model.save_weights(model_path)

    else:
        # Grid search with cross-validaton on training data set
        keras_model = KerasClassifier(build_fn=build_model, verbose=0)
        param_grid = {
            'kernel_size': [2, 3],
            'regularization': [0.01, 0.1, 0.2],
            'weight_constraint': [1., 2., 3.],
            'dropout_prob': [0.2, 0.4, 0.5, 0.6, 0.7],
            'epochs': [12, 20],
            'batch_size': [64, 128, 160]
        }
        grid = GridSearchCV(estimator=keras_model, param_grid=param_grid)
        grid_result = grid.fit(X_train, y_train)

        # Summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))


