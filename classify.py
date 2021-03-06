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
    parser = argparse.ArgumentParser(description='Classify FTDC data.')
    parser.add_argument('model', type=str, help='model file to use')
    parser.add_argument('--local', action='store_true', help='use local mongod for input data')
    parser.add_argument('--prob', action='store_true', help='print per-sample classification probability')
    parser.add_argument('--database', '-d', type=str, help='database name', default='test')
    parser.add_argument('--collection', '-c', type=str, help='collection name', default='test_ftdc')
    parser.add_argument('--stat', action='store_true', help='print stats')
    parser.add_argument('--full', action='store_true', help='print all classification results')
    parser.add_argument('--feat', type=int, help='print top features')
    parser.add_argument('--csv', type=str, help='export as csv')
    parser.add_argument('--delta', action='store_true', help='use first derivative preprocessing')
    args = parser.parse_args()

    ### Load RF model
    model = cPickle.load(open(args.model))
    rf_model = model['classifier']
    ss_coder = model['coder']
    print '\nClasses:'
    print rf_model.classes_

    ### Print top features
    if args.feat:
        print '\nTop features:'
        for x in sorted(zip(rf_model.feature_importances_, ss_coder.feature_names_), reverse=True)[:args.feat]:
            print("%0.4f  %s" % x)
        sys.exit(0)

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
    ftdc_raw = [(x['ftdc'], x.get('class'), x['ts']) for x in conn[args.database][args.collection].find()]
    ftdc = [x[0] for x in ftdc_raw]
    target_classes = [x[1] for x in ftdc_raw]
    timestamps = [str(x[2]) for x in ftdc_raw]
    print 'Len ftdc:', len(ftdc_raw)

    if args.delta:
        ftdc = derivative(ftdc)

    ### Transform input data
    ss_coded = ss_coder.transform(ftdc)

    ### Classify
    print '\nPrediction'
    if args.full:
        for x in zip(timestamps, rf_model.predict(ss_coded)):
            print '{0} {1}'.format(x[0].ljust(27), x[1])
    else:
        curr_class = None
        prev_ts = None
        frame_count = 0
        for x in zip(timestamps, rf_model.predict(ss_coded)):
            frame_count += 1
            if curr_class == None:
                print '{0}'.format(x[0].ljust(27)),
                curr_class = x[1]
            if x[1] != curr_class:
                print '{0}'.format(prev_ts.ljust(27)),
                print '{0}'.format(str(frame_count).rjust(7)),
                print '{0}'.format(curr_class)
                print '{0}'.format(x[0].ljust(27)),
                curr_class = x[1]
                frame_count = 0
            prev_ts = x[0]
        print '{0} {1} {2}'.format(prev_ts.ljust(27), str(frame_count).rjust(7), curr_class)

    ### Dump CSV
    if args.csv:
        with open(args.csv, 'w') as f:
            for x in zip(timestamps, rf_model.predict(ss_coded)):
                f.write('{0},{1}\n'.format(x[0].ljust(27), x[1]))

    ### Classification probabilities
    if args.prob:
        print '\nPrediction probabilities'
        for x in zip(timestamps, rf_model.predict_proba(ss_coded)):
            print '{0} {1}'.format(x[0].ljust(27),x[1])

    if args.stat:
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
