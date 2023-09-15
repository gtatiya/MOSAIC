# Author: Gyan Tatiya

import argparse
import os
import logging
import pickle
from datetime import datetime

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from mosaic.utils import get_config, save_config
from mosaic.dataset import module
from mosaic.models import common, utils


def model_eval(model, dataloader, path, dir):

    model.eval()
    dataset_size = len(dataloader)

    running_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, uni_embeds, info = model.common_step(batch, batch_idx)

            running_loss += loss.item()
            
            for i, uni_embed in enumerate(uni_embeds):
                uni_embed = uni_embed.cpu().numpy()
                trial_data_path = os.sep.join([path, dir, info['object_name'][i]])
                os.makedirs(trial_data_path, exist_ok=True)
                output_file = open(trial_data_path + os.sep + info['trial_num'][i] + '.bin', 'wb')
                pickle.dump(uni_embed, output_file)
                output_file.close()
            
    loss = running_loss / dataset_size

    return loss


if __name__ == '__main__':
    '''
    This script learns unified representations for all modalities under 2 settings.
    Save the unified representations and evaluate it on a task.
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
    parser.add_argument('-setting',
                        choices=['setting1', 'setting2'],
                        default='setting1',
                        help='setting')
    parser.add_argument('-num-folds',
                        default=5,
                        type=int,
                        help='number of folds')
    parser.add_argument('-skip-existing-training',
                        action='store_true',
                        help='skip existing training')
    parser.add_argument('-test-ckpt',
                        choices=['best', 'last'],
                        default='last',
                        help='test checkpoint')
    args = parser.parse_args()

    binary_dataset_path = r'data' + os.sep + args.dataset + '_Binary'
    dir_to_save_unify_repr = 'learned_repr_' + args.test_ckpt
    main_task = 'unify_repr'

    time_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    results_path = 'results' + os.sep + main_task + os.sep + args.setting + os.sep
    os.makedirs(results_path, exist_ok=True)

    log_filepath = results_path + time_stamp + '.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()])

    config = get_config(r'configs' + os.sep + 'dataset_config.yaml')
    model_config = get_config(r'configs' + os.sep + 'model_config.yaml')
    config.update(vars(args))
    config.update(model_config)

    robots = list(config[args.dataset].keys())
    behaviors = config[args.dataset][args.robot]['behaviors']
    modalities_fps = config[args.dataset][args.robot]['modalities_fps']
    modalities_train = config[args.dataset][args.robot]['settings'][args.setting]
    modalities = list(modalities_fps.keys())
    for modality in modalities:
        if modality not in modalities_train:
             modalities_fps.pop(modality, None)

    data_file_path = os.sep.join([binary_dataset_path, 'dataset_metadata.bin'])
    bin_file = open(data_file_path, 'rb')
    metadata = pickle.load(bin_file)
    bin_file.close()

    objects = sorted(list(metadata[behaviors[0]]['objects']))
    objects.remove('no_object')
    trials = sorted(list(metadata[behaviors[0]]['trials']))

    # Hyper-Parameters
    training_epochs = model_config[main_task]['training_epochs']
    learning_rate = model_config[main_task]['learning_rate']
    batch_size = model_config[main_task]['batch_size']
    use_mha = False
    if args.setting == 'setting2':
        use_mha = True

    config_path = r'configs' + os.sep + 'objects_splits' + os.sep + args.dataset + os.sep
    folds_objects_split = get_config(config_path + 'objects_splits.yaml')

    for fold in sorted(folds_objects_split, reverse=False):
        logging.info('fold: {}'.format(fold))

        for behavior in sorted(behaviors, reverse=False):
            logging.info('behavior: {}'.format(behavior))

            save_path = results_path + fold + os.sep + behavior + os.sep
            if args.skip_existing_training and os.path.exists(save_path):
                logging.info('save_path already exists: {}'.format(save_path))
                continue
            else:
                os.makedirs(save_path, exist_ok=True)

            if not utils.have_all_learned_repr(save_path, objects, trials, dir_to_save_unify_repr):

                dataset_module = module.CY101DataModule(binary_dataset_path, metadata, behavior, modalities_fps=modalities_fps)
                dataset_module.prepare_fold(folds_objects_split[fold]['train'], folds_objects_split[fold]['test'])

                model = common.UnifyRepresentations(dataset_module, modalities_train, batch_size=batch_size,
                                                    learning_rate=learning_rate, use_mha=use_mha)
                
                # Initialize the Trainer
                checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', filename='best-{val_loss:.2f}',
                                                      every_n_epochs=1, save_top_k=1, save_last=True)
                logger = TensorBoardLogger(save_dir=save_path, version='', name='')
                trainer = pl.Trainer(max_epochs=training_epochs, callbacks=[checkpoint_callback], logger=logger,
                                     log_every_n_steps=10)

                ckpt_path = os.path.join(trainer.log_dir, 'checkpoints')
                best_ckpt_filepath, last_ckpt_filepath = utils.get_ckpt_filepath(ckpt_path)
                
                # Training
                if last_ckpt_filepath:
                    utils.check_and_fix_ckpt_paths(last_ckpt_filepath, ckpt_path)

                    trainer.fit(model, datamodule=dataset_module, ckpt_path=last_ckpt_filepath)
                else:
                    trainer.fit(model, datamodule=dataset_module)

                best_ckpt_filepath = checkpoint_callback.best_model_path
                last_ckpt_filepath = checkpoint_callback.last_model_path
                
                # Testing using pretrained model
                ckpt_filepath = best_ckpt_filepath
                if args.test_ckpt == 'last':
                    ckpt_filepath = last_ckpt_filepath

                model = common.UnifyRepresentations.load_from_checkpoint(checkpoint_path=best_ckpt_filepath,
                                                                         dataset_module=dataset_module)

                logging.info('TEST RESULTS: ')
                test_dataloader = dataset_module.test_dataloader()
                test_loss = model_eval(model, test_dataloader, save_path, dir_to_save_unify_repr)
                logging.info('test_loss: {}'.format(test_loss))

                logging.info('TRAIN RESULTS: ')
                train_dataloader = dataset_module.train_dataloader()
                train_loss = model_eval(model, train_dataloader, save_path, dir_to_save_unify_repr)
                logging.info('train_loss: {}'.format(train_loss))

    save_config(config, results_path, 'config.yaml')
