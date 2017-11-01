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
from pprint import pprint


def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    '''https://stackoverflow.com/a/26853961/5619724'''
    z = x.copy()
    z.update(y)
    return z

def derivative(data):
    deltas = []
    delta = {}
    for key in data[0]:
        delta['delta_' + key] = 0
    deltas.append(delta)
    for idx in range(1, len(data)):
        delta = {}
        for key in data[idx]:
            val1, val2 = data[idx].get(key), data[idx-1].get(key)
            if val1 is None or val2 is None:
                continue
            delta['delta_' + key] = val1 - val2
        deltas.append(delta)
    for idx in range(len(data)):
        data[idx] = merge_two_dicts(data[idx], deltas[idx])
    return data

if __name__ == '__main__':
    ### Parse argument
    parser = argparse.ArgumentParser(description='Train RF model for FTDC data.')
    parser.add_argument('--local', action='store_true', help='use local mongod for training')
    parser.add_argument('--save', type=str, help='save the generated model to a file')
    parser.add_argument('--database', '-d', type=str, help='training database name', default='test')
    parser.add_argument('--collection', '-c', type=str, help='training collection name', default='test_ftdc')
    parser.add_argument('--delta', action='store_true', help='use first derivative preprocessing')
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
    ftdc_raw = [(x['ftdc'], x['class']) for x in conn[args.database][args.collection].find({'class': {'$exists': True}})]
    ftdc = [x[0] for x in ftdc_raw]
    target_classes = [x[1] for x in ftdc_raw]
    print 'Len ftdc:', len(ftdc_raw)
    print set(target_classes)

    if args.delta:
        ftdc = derivative(ftdc)

    ### Get feature vector
    ss_coder = feature_extraction.DictVectorizer(sparse=True)
    ss_coder.fit_transform(ftdc[0])
    ss_coded = ss_coder.transform(ftdc)

    ### Train RF
    rf_model = ensemble.RandomForestClassifier(n_estimators=20)
    train_size = 0.75
    test_size = 0.25
    ss_coded_train, ss_coded_test, target_classes_train, target_classes_test = model_selection.train_test_split(ss_coded, target_classes, test_size=test_size, train_size=train_size)
    rf_model.fit(ss_coded_train, target_classes_train)
    print 'Training data size:', len(target_classes_train), 'Test data size:', len(target_classes_test)
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
    for x in sorted(zip(rf_model.feature_importances_, ss_coder.feature_names_), reverse=True)[:30]:
        print("%0.4f  %s" % x)

    ### Model data structure
    model = {}
    model['classifier'] = rf_model
    model['coder'] = ss_coder

    ### Save model
    if args.save:
        cPickle.dump(model, open(args.save, 'wb'))
