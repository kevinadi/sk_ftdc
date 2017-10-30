#!/usr/bin/env python
'''
ftdc data format:
{
    _id: ...
    ftdc: {
        ss: ..
        ss: ..
        ss: ..
    },
    class: 'ok' / 'not_ok' / 'problem_class_1'
    source: 1,
    ts: ...
}

Required environment variable:
SK_FTDC_USER: Atlas username
SK_FTDC_PWD: Atlas password
'''

import os
import sys
import pymongo
from pprint import pprint
from sklearn import metrics
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import feature_extraction
from sklearn import model_selection


if __name__ == '__main__':
    ### Get ftdc data
    atlas_username = os.environ.get('SK_FTDC_USER')
    atlas_pwd = os.environ.get('SK_FTDC_PWD')
    conn = pymongo.MongoClient('mongodb://{user}:{pwd}@cluster0-shard-00-00-isvie.mongodb.net:27017,cluster0-shard-00-01-isvie.mongodb.net:27017,cluster0-shard-00-02-isvie.mongodb.net:27017/admin?replicaSet=Cluster0-shard-0&authSource=admin&ssl=true'.format(user=atlas_username, pwd=atlas_pwd))
    ftdc_raw = [(x['ftdc'], x['class']) for x in conn.test.test_ftdc.find()]
    ftdc = [x[0] for x in ftdc_raw]
    target_classes = [x[1] for x in ftdc_raw]
    print 'Len ftdc:', len(ftdc_raw)
    print target_classes

    ### Get feature vector
    ss_coder = feature_extraction.DictVectorizer(sparse=True)
    ss_coder.fit_transform(ftdc[0])
    ss_coded = ss_coder.transform(ftdc)
    print 'ss_coded:', ss_coded.shape

    ### Train RF
    rf_model = ensemble.RandomForestClassifier(n_estimators=20)
    ss_coded_train, ss_coded_test, target_classes_train, target_classes_test = model_selection.train_test_split(ss_coded, target_classes)
    rf_model.fit(ss_coded_train, target_classes_train)
    print 'target classes train:', len(target_classes_train), 'target classes test:', len(target_classes_test)
    print rf_model

    ### Print classification report
    expected = target_classes_test
    predicted = rf_model.predict(ss_coded_test)

    ### Confusion matrix
    print metrics.confusion_matrix(target_classes_test, predicted, labels=None, sample_weight=None)

    ### Print rf_model
    print metrics.classification_report(expected, predicted)

    ### Print cross-validation score
    cv = 3
    scores = model_selection.cross_val_score(rf_model, ss_coded, target_classes, cv=cv)
    print("%d-fold cross-validation accuracy: %0.2f (+/- %0.2f)" % (cv, scores.mean(), scores.std() * 2))

    ### Print top features
    print 'Top features:'
    for x in sorted(zip(rf_model.feature_importances_, ss_coder.feature_names_), reverse=True)[:10]:
        print("%0.4f  %s" % x)
