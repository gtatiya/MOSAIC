# Author: Gyan Tatiya

import argparse
import csv
import os
import logging
import pickle
from datetime import datetime

import numpy as np
from numpy import linalg as LA
import pandas as pd

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import AutoTokenizer, CLIPTextModelWithProjection

from mosaic.utils import get_config, update_all_modalities, update_all_behaviors_modalities, \
                           compute_mean_accuracy, save_config
from mosaic.dataset import module
from mosaic.models import common, utils


def eval_zero_shot_classifier(objects, trials, path, text_embeds, category_ids):

    all_y = []
    all_y_hat = []
    all_proba = []
    for object_name in sorted(objects):
        for trial_num in sorted(trials):

            category = object_name.split('_')[0]
            trial_data_filepath = os.sep.join([path, object_name, trial_num + '.bin'])

            bin_file = open(trial_data_filepath, 'rb')
            unify_repr = pickle.load(bin_file)
            bin_file.close()

            unify_repr /= LA.norm(unify_repr, axis=-1, keepdims=True)
            similarity = text_embeds @ unify_repr.T

            # Normalizing similarity
            similarity_sum = np.sum(similarity, axis=0)  # sum of similarity
            probability = similarity / similarity_sum
            y_hat = np.argmax(probability, axis=0)
            
            all_y.append(category_ids[category])
            all_y_hat.append(y_hat)
            all_proba.append(probability)
    
    y = np.array(all_y)
    y_hat = np.array(all_y_hat)
    proba = np.array(all_proba)
    acc = np.mean(y == y_hat)

    return acc, y, y_hat, proba


def get_behaviors_similarity(path, behaviors, behaviors_similarity, dir, object_name, text_embeds):

    behaviors_similarity_unify_repr = {}
    for behavior in behaviors:
        if behavior not in behaviors_similarity:
            continue
        read_unify_repr_path = os.path.join(path, behavior, dir, object_name)
        trial_num = np.random.choice(os.listdir(read_unify_repr_path))
        trial_data_filepath = read_unify_repr_path + os.sep + trial_num

        bin_file = open(trial_data_filepath, 'rb')
        unify_repr = pickle.load(bin_file)
        bin_file.close()

        unify_repr /= LA.norm(unify_repr, axis=-1, keepdims=True)
        similarity = text_embeds @ unify_repr.T
        behaviors_similarity_unify_repr[behavior] = similarity * behaviors_similarity[behavior]
    
    return behaviors_similarity_unify_repr


if __name__ == '__main__':
    '''
    This script evaluates learned unified representations on several tasks.
    Train/test split: 4 objects in trainset and 1 object in testset for 20 object categories.
    '''

    parser = argparse.ArgumentParser(description='Learn unified representations.')
    parser.add_argument('-dataset',
                        choices=['cy101'],
                        # required=True,
                        default='cy101',
                        help='dataset name')
    parser.add_argument('-robot',
                        choices=['barrett'],
                        default='barrett',
                        help='robot name')
    parser.add_argument('-task',
                        choices=['classify_category', 'fetch_object_task'],
                        default='classify_category',
                        help='task')
    parser.add_argument('-task2',
                        choices=['l-1_d-1_p-0_q-20.yaml', 'l-2_d-1_p-1_q-20.yaml',
                                 'l-3_d-1_p-2_q-20.yaml', 'l-4_d-2_p-2_q-20.yaml',
                                 'l-5/Color_d-1_p-1_q-20.yaml', 'l-5/Deformability_d-1_p-1_q-20.yaml', 'l-5/Hardness_d-1_p-1_q-20.yaml',
                                 'l-5/Material_d-1_p-1_q-20.yaml', 'l-5/Object_state_d-1_p-1_q-20.yaml', 'l-5/Shape_d-1_p-1_q-20.yaml',
                                 'l-5/Size_d-1_p-1_q-20.yaml', 'l-5/Transparency_d-1_p-1_q-20.yaml',
                                 'l-5/Usage_d-1_p-1_q-20.yaml', 'l-5/Weight_d-1_p-1_q-20.yaml'],
                        default='l-1_d-1_p-0_q-20.yaml',
                        help='task2')
    parser.add_argument('-task3',
                        action='store_true',
                        help='use_behaviors_similarity')
    parser.add_argument('-task4',
                        default=0,
                        type=int,
                        help='use_n_behaviors')
    parser.add_argument('-task5',
                        action='store_true',
                        help='use_top_behaviors')
    parser.add_argument('-setting',
                        choices=['setting1', 'setting2'],
                        default='setting1',
                        help='setting')
    parser.add_argument('-save-model',
                        action='store_true',
                        help='save model for classify_category')
    parser.add_argument('-test-ckpt',
                        choices=['best', 'last'],
                        default='last',
                        help='test checkpoint')
    args = parser.parse_args()

    np.random.seed(0)

    binary_dataset_path = r'data' + os.sep + args.dataset + '_Binary'
    dir_to_save_unify_repr = 'learned_repr_' + args.test_ckpt
    main_task = 'unify_repr'

    results_path = 'results' + os.sep + main_task + '_tasks' + os.sep + args.setting + os.sep + args.task + os.sep
    os.makedirs(results_path, exist_ok=True)
    unify_repr_path = 'results' + os.sep + main_task + os.sep + args.setting + os.sep

    log_filepath = r'logs' + os.sep + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    os.makedirs(r'logs', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()])

    config = get_config(r'configs' + os.sep + 'dataset_config.yaml')
    model_config = get_config(r'configs' + os.sep + 'model_config.yaml')
    config.update(vars(args))
    config.update(model_config)

    robots = list(config[args.dataset].keys())
    behaviors = config[args.dataset][args.robot]['behaviors']

    if args.task == 'fetch_object_task':
        use_behaviors_similarity = args.task3
        if args.task4:
            use_n_behaviors = args.task4
        else:
            use_n_behaviors = len(behaviors)
        if args.task5:
            use_top_behaviors = True
        else:
            use_top_behaviors = False
        
        '''
        if use_behaviors_similarity==True: always top n behaviors are used
        if use_top_behaviors==True: only top n behaviors are used otherwise n random behaviors are used
        '''
        
        results_path = 'results' + os.sep + main_task + '_tasks' + os.sep + args.setting + os.sep + args.task + os.sep + \
            ((f'top-{use_n_behaviors}-behaviors_similarity' if use_behaviors_similarity else f'top-{use_n_behaviors}-behaviors' if use_top_behaviors else f'random-{use_n_behaviors}-behaviors') if len(behaviors) > 1 else '') + os.sep
        if args.task2.split('/')[0] == 'l-5':
            os.makedirs(results_path + 'l-5' + os.sep, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)
    
    data_file_path = os.sep.join([binary_dataset_path, 'dataset_metadata.bin'])
    bin_file = open(data_file_path, 'rb')
    metadata = pickle.load(bin_file)
    bin_file.close()

    objects = sorted(list(metadata[behaviors[0]]['objects']))
    objects.remove('no_object')
    trials = sorted(list(metadata[behaviors[0]]['trials']))

    config_path = r'configs' + os.sep + 'objects_splits' + os.sep + args.dataset + os.sep
    folds_objects_split = get_config(config_path + 'objects_splits.yaml')

    if args.task == 'classify_category':

        # Hyper-Parameters
        training_epochs = model_config[args.task]['training_epochs']
        learning_rate = model_config[args.task]['learning_rate']
        batch_size = model_config[args.task]['batch_size']

        folds_behaviors_proba_score = {}
        for fold in sorted(folds_objects_split, reverse=False):
            logging.info('fold: {}'.format(fold))
            folds_behaviors_proba_score.setdefault(fold, {})

            for behavior in sorted(behaviors, reverse=False):
                logging.info('behavior: {}'.format(behavior))
                folds_behaviors_proba_score[fold].setdefault(behavior, {})
                folds_behaviors_proba_score[fold][behavior].setdefault(main_task, {})

                save_path = results_path + fold + os.sep + behavior + os.sep
                
                results_filepath = save_path + 'results_' + args.test_ckpt + '.bin'
                if not os.path.exists(results_filepath):
                    read_unify_repr_path = unify_repr_path + fold + os.sep + behavior + os.sep + dir_to_save_unify_repr
                    dataset_module = module.CY101DataModuleUnifyRepr(read_unify_repr_path, metadata, behavior)
                    dataset_module.prepare_fold(folds_objects_split[fold]['train'], folds_objects_split[fold]['test'])
                    output_size = len(dataset_module.dataset.metadata['category_ids'])

                    model = common.CategoryClassifier(dataset_module, output_size=output_size, batch_size=batch_size,
                                                      learning_rate=learning_rate)
                    
                    if args.save_model:
                        checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max', filename='best-{val_acc:.2f}',
                                                              every_n_epochs=1, save_top_k=1, save_last=True)
                        logger = TensorBoardLogger(save_dir=save_path, version='', name='')
                        trainer = pl.Trainer(max_epochs=training_epochs, callbacks=[checkpoint_callback], logger=logger,
                                             log_every_n_steps=10)
                        
                        ckpt_path = os.path.join(trainer.log_dir, 'checkpoints')
                        best_ckpt_filepath, last_ckpt_filepath = utils.get_ckpt_filepath(ckpt_path)
                    else:
                        logger = TensorBoardLogger(save_dir=save_path, version='', name='')
                        trainer = pl.Trainer(max_epochs=training_epochs, logger=logger, log_every_n_steps=10, enable_checkpointing=False)

                    # Training
                    if args.save_model and last_ckpt_filepath:
                        utils.check_and_fix_ckpt_paths(last_ckpt_filepath, ckpt_path)

                        trainer.fit(model, datamodule=dataset_module, ckpt_path=last_ckpt_filepath)
                    else:
                        trainer.fit(model, datamodule=dataset_module)
                    
                    if args.save_model:
                        best_ckpt_filepath = checkpoint_callback.best_model_path
                        last_ckpt_filepath = checkpoint_callback.last_model_path
                        
                        # Testing using pretrained model
                        ckpt_filepath = best_ckpt_filepath
                        if args.test_ckpt == 'last':
                            ckpt_filepath = last_ckpt_filepath

                        model = common.CategoryClassifier.load_from_checkpoint(checkpoint_path=best_ckpt_filepath,
                                                                               dataset_module=dataset_module)
                                        
                    logging.info('TEST RESULTS: ')
                    test_dataloader = dataset_module.test_dataloader()
                    test_loss, test_acc, test_y, test_y_hat, test_proba = utils.eval_classifier(model, test_dataloader)
                    logging.info('test_acc: {}'.format(test_acc))
                    logging.info('test_y: {}, {}'.format(test_y.shape, test_y))
                    logging.info('test_y_hat: {}, {}'.format(test_y_hat.shape, test_y_hat))
                    folds_behaviors_proba_score[fold][behavior][main_task]['proba'] = test_proba
                    folds_behaviors_proba_score[fold][behavior][main_task]['test_acc'] = test_acc

                    logging.info('TRAIN RESULTS: ')
                    train_dataloader = dataset_module.train_dataloader()
                    train_loss, train_acc, train_y, train_y_hat, train_proba = utils.eval_classifier(model, train_dataloader)
                    logging.info('train_acc: {}'.format(train_acc))
                    folds_behaviors_proba_score[fold][behavior][main_task]['train_acc'] = train_acc

                    # Saving results
                    results = {'proba': test_proba, 'y': test_y, 'test_acc': test_acc, 'train_acc': train_acc}
                    output_file = open(results_filepath, 'wb')
                    pickle.dump(results, output_file)
                    output_file.close()
                else:
                    logging.info('Loading results from: {}'.format(results_filepath))
                    bin_file = open(results_filepath, 'rb')
                    results = pickle.load(bin_file)
                    bin_file.close()

                    folds_behaviors_proba_score[fold][behavior][main_task]['proba'] = results['proba']
                    folds_behaviors_proba_score[fold][behavior][main_task]['test_acc'] = results['test_acc']
                    folds_behaviors_proba_score[fold][behavior][main_task]['train_acc'] = results['train_acc']
                    test_y = results['y']
                
                folds_behaviors_proba_score[fold][behavior] = update_all_modalities(folds_behaviors_proba_score[fold][behavior], test_y)
            
            folds_behaviors_proba_score[fold] = update_all_behaviors_modalities(folds_behaviors_proba_score[fold], test_y)
        
        behaviors_modalities_score = compute_mean_accuracy(folds_behaviors_proba_score, vary_objects=False)

        for behavior in behaviors_modalities_score:
            logging.info('{}: {}'.format(behavior, behaviors_modalities_score[behavior]))
        
        row = ['behavior', main_task]
        
        df = pd.DataFrame(columns=row)
        for behavior in behaviors_modalities_score:

            if not behavior.startswith('all_behaviors_modalities'):
                row = {'behavior': behavior}
                for modality in behaviors_modalities_score[behavior]:
                    value = str(round(behaviors_modalities_score[behavior][modality]['mean'] * 100, 2))
                    value += '±' + str(round(behaviors_modalities_score[behavior][modality]['std'] * 100, 2))
                    row[modality] = value
                df = pd.concat([df, pd.DataFrame.from_records([row])])

        logging.info('df:\n{}'.format(df))
        results_file_path = results_path + args.task + '_results_' + args.test_ckpt + '.csv'
        df.to_csv(results_file_path, index=False)

        with open(results_file_path, 'a') as f:
            writer = csv.writer(f, lineterminator="\n")

            row = ['Average: ']
            for column in df:
                if column != 'behavior':
                    values = [float(value[0].split('±')[0]) for value in df[[column]].values.tolist()]
                    row.append(str(round(np.mean(values), 2)) + '±' + str(round(np.std(values), 2)))
            writer.writerow(row)

            writer.writerow(['all_behaviors_modalities: ',
                            str(round(behaviors_modalities_score['all_behaviors_modalities']['mean'] * 100, 2)) + '±' + 
                            str(round(behaviors_modalities_score['all_behaviors_modalities']['std'] * 100, 2)),
                            'all_behaviors_modalities_train: ',
                            str(round(behaviors_modalities_score['all_behaviors_modalities_train']['mean'] * 100, 2)) + '±' +
                            str(round(behaviors_modalities_score['all_behaviors_modalities_train']['std'] * 100, 2)),
                            'all_behaviors_modalities_test: ',
                            str(round(behaviors_modalities_score['all_behaviors_modalities_test']['mean'] * 100, 2)) + '±' +
                            str(round(behaviors_modalities_score['all_behaviors_modalities_test']['std'] * 100, 2))])
    
    elif args.task == 'fetch_object_task':

        task_path = r'configs' + os.sep + 'fetch_object_task' + os.sep
        task_filename = args.task2
        task_name, fileext = os.path.splitext(task_filename)
        folds_queries = get_config(task_path + task_filename)
        logging.info('task_name: {}'.format(task_name))

        # Load CLIPText model and tokenizer
        text_model = CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-base-patch32')
        text_model.eval()
        text_tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-base-patch32')

        behaviors_text_embeds = {}
        for behavior in behaviors:
            x = text_tokenizer(['Perform ' + behavior + ' action'], padding=True, return_tensors='pt')['input_ids']
            text_embeds = text_model(x).text_embeds[0].detach().numpy()
            text_embeds /= LA.norm(text_embeds, axis=-1, keepdims=True)
            behaviors_text_embeds[behavior] = text_embeds
        
        folds_acc = {}
        for fold in sorted(folds_objects_split, reverse=False):
            logging.info('fold: {}'.format(fold))
            folds_acc.setdefault(fold, {})

            total_target_score = 0
            total_distractors_score = {}
            for task in folds_queries[fold]:
                target_object_name = task['target_object_name']
                target_query = task['target_query']
                target_sound_behavior = task['target_sound_behavior']

                x = text_tokenizer([target_query], padding=True, return_tensors='pt')['input_ids']
                text_embeds = text_model(x).text_embeds[0].detach().numpy()
                text_embeds /= LA.norm(text_embeds, axis=-1, keepdims=True)

                behaviors_similarity = {}
                for behavior in behaviors_text_embeds:
                    if use_behaviors_similarity:
                        similarity = text_embeds @ behaviors_text_embeds[behavior].T
                        behaviors_similarity[behavior] = similarity
                    else:
                        behaviors_similarity[behavior] = 1

                if use_behaviors_similarity:
                    behaviors_similarity = {behavior: similarity for behavior, similarity in sorted(behaviors_similarity.items(), key=lambda item: item[1], reverse=True)[:use_n_behaviors]}
                
                    factor = 1.0/sum(behaviors_similarity.values())
                    behaviors_similarity = {k: v*factor for k, v in behaviors_similarity.items()}

                target_behaviors_similarity = get_behaviors_similarity(unify_repr_path + fold, behaviors, behaviors_similarity, dir_to_save_unify_repr, target_object_name, text_embeds)

                if not use_behaviors_similarity:
                    target_behaviors_similarity = [(behavior, similarity) for behavior, similarity in sorted(target_behaviors_similarity.items(), key=lambda item: item[1], reverse=True)]
                    if use_top_behaviors:
                        target_behaviors_similarity = target_behaviors_similarity[:use_n_behaviors]
                    else:
                        np.random.shuffle(target_behaviors_similarity)
                        target_behaviors_similarity = target_behaviors_similarity[:use_n_behaviors]
                    target_behaviors_similarity = {behavior: similarity for behavior, similarity in target_behaviors_similarity}
                
                target_score = sum(target_behaviors_similarity.values())

                distractors_score = []
                for d_idx, distractor_info in enumerate(task['distractor_objects']):
                    total_distractors_score.setdefault(d_idx, 0)
                    distractor_object = distractor_info['distractor_object']
                    distractor_object_query = distractor_info['distractor_object_query']

                    distractor_behaviors_similarity = get_behaviors_similarity(unify_repr_path + fold, behaviors, behaviors_similarity, dir_to_save_unify_repr, distractor_object, text_embeds)
                    if not use_behaviors_similarity:
                        if use_top_behaviors:
                            distractor_behaviors_similarity = [(behavior, similarity) for behavior, similarity in sorted(distractor_behaviors_similarity.items(), key=lambda item: item[1], reverse=True)]
                            distractor_behaviors_similarity = distractor_behaviors_similarity[:use_n_behaviors]
                            distractor_behaviors_similarity = {behavior: similarity for behavior, similarity in distractor_behaviors_similarity}
                        else:
                            # Use the same random behaviors selected for target_behaviors_similarity
                            distractor_behaviors_similarity = {behavior: distractor_behaviors_similarity[behavior] for behavior in target_behaviors_similarity}
                    
                    distractor_score = sum(distractor_behaviors_similarity.values())
                    distractors_score.append(distractor_score)
                                                    
                index_max = np.argmax(distractors_score)
                
                if target_score > distractors_score[index_max]:
                    total_target_score += 1
                else:
                    total_distractors_score[index_max] += 1
            
            logging.info('total_target_score: {}'.format(total_target_score))
            logging.info('total_distractors_score: {}'.format(total_distractors_score))

            total_distractor_score = sum(total_distractors_score.values())
            total_score = total_target_score + total_distractor_score
            logging.info('total_score: {}'.format(total_score))
            assert total_score == len(folds_queries[fold]), 'total_score is not correct'

            # Computing Accuracy
            target_acc = total_target_score / total_score
            distractors_acc = {}
            for d_idx in total_distractors_score:
                distractors_acc[d_idx] = total_distractors_score[d_idx] / total_score
            total_distractor_acc = sum(distractors_acc.values())
            total_acc = target_acc + total_distractor_acc
            assert round(total_acc) == 1, f'total_acc ({total_acc}) is not correct'

            logging.info('target_acc: {}'.format(target_acc))
            logging.info('distractors_acc: {}'.format(distractors_acc))

            folds_acc[fold]['target_acc'] = target_acc
            folds_acc[fold]['distractors_acc'] = distractors_acc

        # Computing mean accuracy of all folds
        folds_target_acc = []
        folds_distractors_acc = {}
        for fold in folds_acc:
            folds_target_acc.append(folds_acc[fold]['target_acc'])
            for d_idx in folds_acc[fold]['distractors_acc']:
                folds_distractors_acc.setdefault(d_idx, [])
                folds_distractors_acc[d_idx].append(folds_acc[fold]['distractors_acc'][d_idx])
        logging.info('folds_target_acc: {}'.format(folds_target_acc))
        logging.info('folds_distractors_acc: {}'.format(folds_distractors_acc))

        folds_target_acc = np.mean(folds_target_acc)
        for d_idx in folds_distractors_acc:
            folds_distractors_acc[d_idx] = np.mean(folds_distractors_acc[d_idx])        
        logging.info('folds_target_acc: {}'.format(folds_target_acc))
        logging.info('folds_distractors_acc: {}'.format(folds_distractors_acc))

        row = ['Object', 'Accuracy']
        df = pd.DataFrame(columns=row)
        row = {'Object': 'Target Object', 'Accuracy': folds_target_acc}
        df = pd.concat([df, pd.DataFrame.from_records([row])])
        for d_idx in folds_distractors_acc:
            row = {'Object': f'Distractor Object {d_idx + 1}', 'Accuracy': folds_distractors_acc[d_idx]}
            df = pd.concat([df, pd.DataFrame.from_records([row])])
        
        logging.info('df:\n{}'.format(df))
        results_file_path = results_path + task_name + '_results_' + args.test_ckpt + '.csv'
        df.to_csv(results_file_path, index=False)

    save_config(config, results_path, 'config.yaml')
