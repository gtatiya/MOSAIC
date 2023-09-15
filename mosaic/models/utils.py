# Author: Gyan Tatiya

import logging
import os
import numpy as np

import torch


def have_all_learned_repr(path, objects, trials, dir):

    for object_name in objects:
        for trial in trials:
            trial_data_filepath = os.sep.join([path, dir, object_name, trial + '.bin'])
            if not os.path.exists(trial_data_filepath):
                return False
    
    return True


def get_ckpt_filepath(ckpt_path: str):
    
    """
    Find the highest accuracy checkpoint path and last checkpoint path in ckpt_path.
    For best, checkpoints should be save in this format: best-val_acc=0.34.ckpt.
    For last, checkpoint should be save as last.ckpt.
    If checkpoint is not found, empty string is returned.
    :param ckpt_path: dir where the checkpoints are being saved.
    :return: path of best and last checkpoints.
    """

    best_acc_ckpt_filename = ''
    best_acc = -np.inf
    last_acc_ckpt_filename = ''
    if os.path.exists(ckpt_path):
        ckpt_files = os.listdir(ckpt_path)
        
        for ckpt_filename in ckpt_files:
            if ckpt_filename.startswith('best'):
                curr_ckpt_acc, fileext = os.path.splitext(ckpt_filename.split('-')[1].split('=')[1])
                if float(curr_ckpt_acc) > best_acc:
                    best_acc = float(curr_ckpt_acc)
                    best_acc_ckpt_filename = ckpt_filename
            elif ckpt_filename.startswith('last'):
                last_acc_ckpt_filename = ckpt_filename
        
        if best_acc_ckpt_filename:
            best_acc_ckpt_filename = os.path.join(ckpt_path, best_acc_ckpt_filename)
        if last_acc_ckpt_filename:
            last_acc_ckpt_filename = os.path.join(ckpt_path, last_acc_ckpt_filename)

    return best_acc_ckpt_filename, last_acc_ckpt_filename


def check_and_fix_ckpt_paths(ckpt_filepath, ckpt_path):
    """
    If ckpt_path is changed, then make sure that paths in ckpt_filepath are the same as ckpt_path
    """

    checkpoint = torch.load(ckpt_filepath, map_location=lambda storage, loc: storage)

    dirpath = ''
    dirpath_changed = False
    for callbackname in checkpoint['callbacks']:
        dirpath = checkpoint['callbacks'][callbackname]['dirpath']
        if dirpath != ckpt_path:
            dirpath_changed = True
            break

    if dirpath_changed:
        logging.info('ckpt_path changed from {} to {}'.format(dirpath, ckpt_path))
        for callbackname in checkpoint['callbacks']:
            dirpath = checkpoint['callbacks'][callbackname]['dirpath']
            for key in checkpoint['callbacks'][callbackname]:
                if key == 'dirpath':
                    checkpoint['callbacks'][callbackname][key] = ckpt_path
                elif 'path' in key:
                    rel = os.path.relpath(checkpoint['callbacks'][callbackname][key], dirpath)
                    checkpoint['callbacks'][callbackname][key] = os.path.join(ckpt_path, rel)
                elif 'models' in key:
                    models = {}
                    for model_filepath in checkpoint['callbacks'][callbackname][key]:
                        rel = os.path.relpath(model_filepath, dirpath)
                        model_filepath_new = os.path.join(ckpt_path, rel)
                        models[model_filepath_new] = checkpoint['callbacks'][callbackname][key][model_filepath]
                    checkpoint['callbacks'][callbackname][key] = models

        torch.save(checkpoint, ckpt_filepath)
    else:
        logging.info('ckpt_path {} and dirpath {} are the same'.format(ckpt_path, dirpath))


def eval_classifier(model, dataloader):

    model.eval()
    dataset_size = len(dataloader)

    running_loss = 0
    running_acc = 0
    all_y = []
    all_y_hat = []
    all_proba = []
    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, acc, y, y_hat, proba = model.common_step(batch, batch_idx)

            running_loss += loss.item()
            running_acc += acc.item()
            all_y.extend(y.cpu().numpy())
            all_y_hat.extend(y_hat.cpu().numpy())
            all_proba.extend(proba.cpu().numpy())
    
    loss = running_loss / dataset_size
    acc = running_acc / dataset_size
    y = np.array(all_y)
    y_hat = np.array(all_y_hat)
    proba = np.array(all_proba)

    return loss, acc, y, y_hat, proba
