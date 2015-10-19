import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys

sys.path.append('D:\\libs\\xgboost\\wrapper')
import xgboost as xgb

print("Load the training/test data using pandas")
train = pd.read_csv("data/training.csv")
test = pd.read_csv("data/test.csv")

print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5])

print("Train a Random Forest model")
rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(train[features], train["signal"])

print("Train a XGBoost model")
params = {'objective': 'binary:logistic',
          'eta': 0.1,
          'max_depth': 5,
          'min_child_weight': 3,
          'silent': 1,
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          'seed': 1}
num_trees = 310
gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)

print("Make predictions on the test set")
test_probs = (rf.predict_proba(test[features])[:, 1] +
              gbm.predict(xgb.DMatrix(test[features]))) / 2
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("submissions/xgb_rf.csv", index=False)