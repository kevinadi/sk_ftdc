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
import argparse
import cPickle


if __name__ == '__main__':
    ### Parse argument
    parser = argparse.ArgumentParser(description='Classify FTDC data.')
    parser.add_argument('model', type=str, help='model file to use')
    parser.add_argument('--local', action='store_true', help='use local mongod for input data')
    args = parser.parse_args()

    ### Connection string
    if args.local:
        connstr = 'mongodb://localhost'
    else:
        atlas_username = os.environ.get('SK_FTDC_USER')
        atlas_pwd = os.environ.get('SK_FTDC_PWD')
        connstr = 'mongodb://{user}:{pwd}@cluster0-shard-00-00-isvie.mongodb.net:27017,cluster0-shard-00-01-isvie.mongodb.net:27017,cluster0-shard-00-02-isvie.mongodb.net:27017/admin?replicaSet=Cluster0-shard-0&authSource=admin&ssl=true'.format(user=atlas_username, pwd=atlas_pwd)
    print 'Connecting to', connstr

    ### Get ftdc data
    conn = pymongo.MongoClient(connstr)
    ftdc_raw = [(x['ftdc'], x['class'], x['ts']) for x in conn.test.test_ftdc.find()]
    ftdc = [x[0] for x in ftdc_raw]
    target_classes = [x[1] for x in ftdc_raw]
    timestamps = [str(x[2]) for x in ftdc_raw]
    print 'Len ftdc:', len(ftdc_raw)

    ### Load RF model
    model = cPickle.load(open(args.model))
    rf_model = model['classifier']
    ss_coder = model['coder']
    print '\nClasses:'
    print rf_model.classes_

    ### Transform input data
    ss_coded = ss_coder.transform(ftdc)

    ### Classify
    print '\nPrediction'
    for x in zip(timestamps, rf_model.predict(ss_coded)):
        print '{0} {1}'.format(x[0].ljust(27),x[1])

    ### Classification probabilities
    print '\nPrediction probabilities'
    for x in zip(timestamps, rf_model.predict_proba(ss_coded)):
        print '{0} {1}'.format(x[0].ljust(27),x[1])

    ### Print classification report
    expected = target_classes
    predicted = rf_model.predict(ss_coded)

    ### Confusion matrix
    print '\nConfusion matrix:'
    print metrics.confusion_matrix(target_classes, predicted, labels=None, sample_weight=None)

    ### Print rf_model
    print metrics.classification_report(expected, predicted)

    ### Print cross-validation score
    cv = 3
    scores = model_selection.cross_val_score(rf_model, ss_coded, target_classes, cv=cv)
    print("%d-fold cross-validation accuracy: %0.2f (+/- %0.2f)" % (cv, scores.mean(), scores.std() * 2))
