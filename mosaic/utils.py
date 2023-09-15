# Author: Gyan Tatiya

import os
import yaml
import logging

import numpy as np


def get_config(config_path):
    with open(config_path) as file:
        return yaml.safe_load(file)


def save_config(config, config_path, config_file, default_flow_style=False):
    os.makedirs(config_path, exist_ok=True)
    with open(config_path + os.sep + config_file, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=default_flow_style)


def combine_probability(proba_acc_list_, y_test_, acc=None):
    # For each classifier, combine weighted probability based on its accuracy score
    proba_list = []
    for proba_acc in proba_acc_list_:
        y_proba = proba_acc['proba']
        if acc and proba_acc[acc] > 0:
            # Multiply the score by probability to combine each classifier's performance accordingly
            # IMPORTANT: This will discard probability when the accuracy is 0
            y_proba = y_proba * proba_acc[acc]  # weighted probability
            proba_list.append(y_proba)
        elif not acc:
            proba_list.append(y_proba)  # Uniform combination, probability is combined even when the accuracy is 0

    # If all the accuracy is 0 in proba_acc_list_, the fill proba_list with chance accuracy
    if len(proba_list) == 0:
        num_examples, num_classes = proba_acc_list_[0]['proba'].shape
        chance_prob = (100 / num_classes) / 100
        proba_list = np.full((1, num_examples, num_classes), chance_prob)

    # Combine weighted probability of all classifiers
    y_proba_norm = np.zeros(len(proba_list[0][0]))
    for proba in proba_list:
        y_proba_norm = y_proba_norm + proba

    # Normalizing probability
    y_proba_norm_sum = np.sum(y_proba_norm, axis=1)  # sum of weighted probability
    y_proba_norm_sum = np.repeat(y_proba_norm_sum, len(proba_list[0][0]), axis=0).reshape(y_proba_norm.shape)
    y_proba_norm = y_proba_norm / y_proba_norm_sum

    y_proba_pred = np.argmax(y_proba_norm, axis=1)
    y_prob_acc = np.mean(y_test_ == y_proba_pred)

    return y_proba_norm, y_prob_acc


def update_all_modalities(modalities_proba_score, y_test_):
    # For each modality, combine weighted probability based on its accuracy score

    proba_acc_list = []
    for modality_ in modalities_proba_score:
        proba_acc = {'proba': modalities_proba_score[modality_]['proba'],
                     'train_acc': modalities_proba_score[modality_]['train_acc'],
                     'test_acc': modalities_proba_score[modality_]['test_acc']}
        proba_acc_list.append(proba_acc)

    y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test_.ravel())
    modalities_proba_score.setdefault('all_modalities', {})
    modalities_proba_score['all_modalities']['proba'] = y_proba_norm
    modalities_proba_score['all_modalities']['test_acc'] = y_prob_acc

    y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test_.ravel(), 'train_acc')
    modalities_proba_score.setdefault('all_modalities_train', {})
    modalities_proba_score['all_modalities_train']['proba'] = y_proba_norm
    modalities_proba_score['all_modalities_train']['test_acc'] = y_prob_acc

    y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test_.ravel(), 'test_acc')
    modalities_proba_score.setdefault('all_modalities_test', {})
    modalities_proba_score['all_modalities_test']['proba'] = y_proba_norm
    modalities_proba_score['all_modalities_test']['test_acc'] = y_prob_acc

    return modalities_proba_score


def update_all_behaviors_modalities(behaviors_modalities_proba_score, y_test_):
    # For each behavior and modality, combine weighted probability based on its accuracy score

    proba_acc_list = []
    for behavior_ in behaviors_modalities_proba_score:
        for modality_ in behaviors_modalities_proba_score[behavior_]:
            if not modality_.startswith('all_modalities'):
                proba_acc = {'proba': behaviors_modalities_proba_score[behavior_][modality_]['proba'],
                             'train_acc': behaviors_modalities_proba_score[behavior_][modality_]['train_acc'],
                             'test_acc': behaviors_modalities_proba_score[behavior_][modality_]['test_acc']}
                proba_acc_list.append(proba_acc)

    y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test_.ravel())
    behaviors_modalities_proba_score.setdefault('all_behaviors_modalities', {})
    behaviors_modalities_proba_score['all_behaviors_modalities']['proba'] = y_proba_norm
    behaviors_modalities_proba_score['all_behaviors_modalities']['test_acc'] = y_prob_acc

    y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test_.ravel(), 'train_acc')
    behaviors_modalities_proba_score.setdefault('all_behaviors_modalities_train', {})
    behaviors_modalities_proba_score['all_behaviors_modalities_train']['proba'] = y_proba_norm
    behaviors_modalities_proba_score['all_behaviors_modalities_train']['test_acc'] = y_prob_acc

    y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test_.ravel(), 'test_acc')
    behaviors_modalities_proba_score.setdefault('all_behaviors_modalities_test', {})
    behaviors_modalities_proba_score['all_behaviors_modalities_test']['proba'] = y_proba_norm
    behaviors_modalities_proba_score['all_behaviors_modalities_test']['test_acc'] = y_prob_acc

    return behaviors_modalities_proba_score


def compute_mean_accuracy(folds_behaviors_modalities_proba_score, acc='test_acc', vary_objects=True,
                          behavior_present=True):

    behaviors_modalities_score = {}
    for fold_ in folds_behaviors_modalities_proba_score:
        if vary_objects:
            for objects_per_label_ in folds_behaviors_modalities_proba_score[fold_]:
                behaviors_modalities_score.setdefault(objects_per_label_, {})
                if behavior_present:
                    for behavior_ in folds_behaviors_modalities_proba_score[fold_][objects_per_label_]:
                        if behavior_.startswith('all_behaviors_modalities'):
                            behaviors_modalities_score[objects_per_label_].setdefault(behavior_, [])
                            y_prob_acc = folds_behaviors_modalities_proba_score[fold_][objects_per_label_][behavior_][acc]
                            behaviors_modalities_score[objects_per_label_][behavior_].append(y_prob_acc)
                        else:
                            behaviors_modalities_score[objects_per_label_].setdefault(behavior_, {})
                            for modality_ in folds_behaviors_modalities_proba_score[fold_][objects_per_label_][behavior_]:
                                behaviors_modalities_score[objects_per_label_][behavior_].setdefault(modality_, [])
                                y_prob_acc = \
                                    folds_behaviors_modalities_proba_score[fold_][objects_per_label_][behavior_][modality_][
                                        acc]
                                behaviors_modalities_score[objects_per_label_][behavior_][modality_].append(y_prob_acc)
                else:
                    for modality_ in folds_behaviors_modalities_proba_score[fold_][objects_per_label_]:
                        behaviors_modalities_score[objects_per_label_].setdefault(modality_, [])
                        y_prob_acc = folds_behaviors_modalities_proba_score[fold_][objects_per_label_][modality_][acc]
                        behaviors_modalities_score[objects_per_label_][modality_].append(y_prob_acc)
        else:
            if behavior_present:
                for behavior_ in folds_behaviors_modalities_proba_score[fold_]:
                    if behavior_.startswith('all_behaviors_modalities'):
                        behaviors_modalities_score.setdefault(behavior_, [])
                        y_prob_acc = folds_behaviors_modalities_proba_score[fold_][behavior_][acc]
                        behaviors_modalities_score[behavior_].append(y_prob_acc)
                    else:
                        behaviors_modalities_score.setdefault(behavior_, {})
                        for modality_ in folds_behaviors_modalities_proba_score[fold_][behavior_]:
                            behaviors_modalities_score[behavior_].setdefault(modality_, [])
                            y_prob_acc = folds_behaviors_modalities_proba_score[fold_][behavior_][modality_][acc]
                            behaviors_modalities_score[behavior_][modality_].append(y_prob_acc)
            else:
                for modality_ in folds_behaviors_modalities_proba_score[fold_]:
                    behaviors_modalities_score.setdefault(modality_, [])
                    y_prob_acc = folds_behaviors_modalities_proba_score[fold_][modality_][acc]
                    behaviors_modalities_score[modality_].append(y_prob_acc)

    if vary_objects:
        for objects_per_label_ in behaviors_modalities_score:
            if behavior_present:
                for behavior_ in behaviors_modalities_score[objects_per_label_]:
                    if behavior_.startswith('all_behaviors_modalities'):
                        behaviors_modalities_score[objects_per_label_][behavior_] = {
                            'mean': np.mean(behaviors_modalities_score[objects_per_label_][behavior_]),
                            'std': np.std(behaviors_modalities_score[objects_per_label_][behavior_])}
                    else:
                        for modality_ in behaviors_modalities_score[objects_per_label_][behavior_]:
                            behaviors_modalities_score[objects_per_label_][behavior_][modality_] = {
                                'mean': np.mean(behaviors_modalities_score[objects_per_label_][behavior_][modality_]),
                                'std': np.std(behaviors_modalities_score[objects_per_label_][behavior_][modality_])}
            else:
                for modality_ in behaviors_modalities_score[objects_per_label_]:
                    behaviors_modalities_score[objects_per_label_][modality_] = {
                        'mean': np.mean(behaviors_modalities_score[objects_per_label_][modality_]),
                        'std': np.std(behaviors_modalities_score[objects_per_label_][modality_])}
    else:
        if behavior_present:
            for behavior_ in behaviors_modalities_score:
                if behavior_.startswith('all_behaviors_modalities'):
                    behaviors_modalities_score[behavior_] = {
                        'mean': np.mean(behaviors_modalities_score[behavior_]),
                        'std': np.std(behaviors_modalities_score[behavior_])}
                else:
                    for modality_ in behaviors_modalities_score[behavior_]:
                        behaviors_modalities_score[behavior_][modality_] = {
                            'mean': np.mean(behaviors_modalities_score[behavior_][modality_]),
                            'std': np.std(behaviors_modalities_score[behavior_][modality_])}
        else:
            for modality_ in behaviors_modalities_score:
                behaviors_modalities_score[modality_] = {
                    'mean': np.mean(behaviors_modalities_score[modality_]),
                    'std': np.std(behaviors_modalities_score[modality_])}

    return behaviors_modalities_score
