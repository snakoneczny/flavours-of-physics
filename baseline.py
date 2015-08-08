import pandas
from sklearn.ensemble import GradientBoostingClassifier
import evaluation

# Read training data
folder = 'data/'
train = pandas.read_csv(folder + 'training.csv', index_col='id')
# print train.head()

# Define training features
variables = ['LifeTime',
             'FlightDistance',
             'pt',
             ]

# Baseline training
baseline = GradientBoostingClassifier(n_estimators=40, learning_rate=0.01, subsample=0.7,
                                      min_samples_leaf=10, max_depth=7, random_state=11)
baseline.fit(train[variables], train['signal'])

# Check agreement test
check_agreement = pandas.read_csv(folder + 'check_agreement.csv', index_col='id')
agreement_probs = baseline.predict_proba(check_agreement[variables])[:, 1]

ks = evaluation.compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)
print 'KS metric', ks, ks < 0.09

# Check correlation test
check_correlation = pandas.read_csv(folder + 'check_correlation.csv', index_col='id')
correlation_probs = baseline.predict_proba(check_correlation[variables])[:, 1]
cvm = evaluation.compute_cvm(correlation_probs, check_correlation['mass'])
print 'CvM metric', cvm, cvm < 0.002

# Compute weighted AUC on the training data with min_ANNmuon > 0.4
train_eval = train[train['min_ANNmuon'] > 0.4]
train_probs = baseline.predict_proba(train_eval[variables])[:, 1]
AUC = evaluation.roc_auc_truncated(train_eval['signal'], train_probs)
print 'AUC', AUC

# Predict test, create file for kaggle
test = pandas.read_csv(folder + 'test.csv', index_col='id')
result = pandas.DataFrame({'id': test.index})
result['prediction'] = baseline.predict_proba(test[variables])[:, 1]

result.to_csv('submissions/baseline.csv', index=False, sep=',')
