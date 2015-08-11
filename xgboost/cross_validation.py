import numpy as np
from sklearn.cross_validation import StratifiedKFold
import pandas
import evaluation
import sys

sys.path.append('D:\\libs\\xgboost\\wrapper')
import xgboost as xgb

# Read training data
folder = '../data/'
train = pandas.read_csv(folder + 'training.csv', index_col='id')

# Define features to drop from train data
# variables_to_drop = ['mass', 'production', 'min_ANNmuon', 'signal', 'SPDhits', 'IP', 'IPSig', ]
variables_to_drop = ['mass', 'production', 'min_ANNmuon', 'signal',
                     'SPDhits', 'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta', ]

# Parameters space creation
params = [[6, 20, 50], [0.3, 0.1, 0.04]]
params_space = []
for i in xrange(len(params[0])):
    for j in xrange(len(params[1])):
        params_space.append([params[0][i], params[1][j]])

# Grid search
grid_errors = []
grid_best_iterations = []
for params in params_space:

    # Cross validation
    skf = StratifiedKFold(train['signal'].values, 8, shuffle=False)
    errors = []
    best_iterations = []
    for i_train, i_test in skf:
        train_cv = train.iloc[i_train]
        test_cv = train.iloc[i_test]

        # test on the data with min_ANNmuon > 0.4
        min_ANNmuon = test_cv['min_ANNmuon'] > 0.4
        test_cv = test_cv[min_ANNmuon]

        # Create data matrices for XGBoost
        train_X = train_cv.drop(variables_to_drop, 1).values
        train_y = train_cv['signal'].values
        test_X = test_cv.drop(variables_to_drop, 1).values
        test_y = test_cv['signal'].values
        xg_train = xgb.DMatrix(train_X, label=train_y)
        xg_test = xgb.DMatrix(test_X, label=test_y)

        # Setup parameters
        param = {'silent': 1, 'nthread': 2, 'objective': 'binary:logistic', 'eval_metric': 'auc',
                 'max_depth': params[0], 'eta': params[1]}  # , 'subsample': 0.5
        n_rounds = 4000  # Just a big number to trigger early stopping and best iteration

        # Train
        xgb_model = xgb.train(param, xg_train, n_rounds, [(xg_train, 'train'), (xg_test, 'test')],
                              early_stopping_rounds=20)
        # Predict
        predictions = xgb_model.predict(xg_test)

        # Compute weighted AUC
        AUC = evaluation.roc_auc_truncated(test_y, predictions)
        errors.append(AUC)
        print 'AUC', AUC

        # Save best iteration
        best_iterations.append(xgb_model.best_iteration)

    # Append new grid error
    grid_errors.append(np.mean(errors))
    grid_best_iterations.append(list(best_iterations))

# Show results
for i in xrange(len(params_space)):
    print "Params: %s, wighted AUC: %f, best iterations: %s, mean: %f" % (
        str(params_space[i]), grid_errors[i], str(grid_best_iterations[i]), np.mean(grid_best_iterations[i]))
