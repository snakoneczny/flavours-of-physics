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
# variables_to_drop = ['mass', 'production', 'min_ANNmuon', 'signal',
#                      'SPDhits', 'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta', ]
variables_to_drop = ['mass', 'production', 'min_ANNmuon', 'signal',
                     'SPDhits', ]


# Train xgb model on train data
train_X = train.drop(variables_to_drop, 1).values
train_y = train['signal'].values
xg_train = xgb.DMatrix(train_X, label=train_y)

# params = {'silent': 1, 'nthread': 2, 'objective': 'binary:logistic', 'eval_metric': 'auc',
#          'max_depth': 6, 'eta': 0.3}

params = {'objective': 'binary:logistic',
          'eta': 0.3,
          'max_depth': 5,
          'min_child_weight': 3,
          'silent': 1,
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          'seed': 1,
          'nthread': 2}
num_trees = 250

n_rounds = 120
watchlist = [(xg_train, 'train')]

xgb_model = xgb.train(params, xg_train, num_trees, watchlist)
# xgb_model = xgb.train(params, xg_train, n_rounds, watchlist)

# Check agreement test
check_agreement = pandas.read_csv(folder + 'check_agreement.csv', index_col='id')
xg_check_agreement = xgb.DMatrix(check_agreement.values)
agreement_probs = xgb_model.predict(xg_check_agreement)

ks = evaluation.compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)
print 'KS metric', ks, ks < 0.09

# Check correlation test
check_correlation = pandas.read_csv(folder + 'check_correlation.csv', index_col='id')
xg_check_correlation = xgb.DMatrix(check_correlation.values)
correlation_probs = xgb_model.predict(xg_check_correlation)
cvm = evaluation.compute_cvm(correlation_probs, check_correlation['mass'])
print 'CvM metric', cvm, cvm < 0.002

# Compute weighted AUC on the training data with min_ANNmuon > 0.4
train_eval = train[train['min_ANNmuon'] > 0.4]
train_eval_X = train_eval.drop(variables_to_drop, 1).values
xg_train_eval = xgb.DMatrix(train_eval_X)
train_probs = xgb_model.predict(xg_train_eval)
AUC = evaluation.roc_auc_truncated(train_eval['signal'], train_probs)
print 'AUC', AUC

# Predict test, create file for kaggle
test = pandas.read_csv(folder + 'test.csv', index_col='id')
test_X = test.values
xg_test = xgb.DMatrix(test_X)
result = pandas.DataFrame({'id': test.index})

result['prediction'] = xgb_model.predict(xg_test)

result.to_csv('../submissions/xgb.csv', index=False, sep=',')
