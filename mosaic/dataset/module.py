# Author: Gyan Tatiya

import os
import pickle
import logging

import numpy as np
from skimage.transform import resize

from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl


class CY101Dataset(Dataset):
    def __init__(self, dataset_path, metadata, behavior, modalities_fps):

        self.metadata = metadata
        self.behavior = behavior
        self.modalities_fps = modalities_fps

        objects = self.metadata[behavior]['objects']
        trials = self.metadata[behavior]['trials']

        self.data_filepaths = []
        self.objects_trials_idx = {}
        self.objects_texts = {}
        i = 0
        for object_name in sorted(objects):
            if object_name in ['no_object']:
                continue
            for trial_num in sorted(trials):
                modality_trial_data = {'object_name': object_name, 'trial_num': trial_num}
                for modality in modalities_fps:
                    trial_data_filepath = ''
                    if modality == 'text':
                        text_data_filepath = os.sep.join([dataset_path, behavior, object_name, 'text.txt'])
                        with open(text_data_filepath, 'r') as f:
                            self.objects_texts[object_name] = [line.rstrip() for line in f]
                    elif modality in ['image', 'video']:
                        trial_data_filepath = os.sep.join([dataset_path, behavior, object_name, trial_num, 'vision.bin'])
                    else:
                        trial_data_filepath = os.sep.join([dataset_path, behavior, object_name, trial_num, modality + '.bin'])
                    modality_trial_data[modality] = trial_data_filepath
                self.data_filepaths.append(modality_trial_data)

                self.objects_trials_idx.setdefault(object_name, [])
                self.objects_trials_idx[object_name].append(i)
                i += 1
        
        self.duration = self.metadata[self.behavior]['modalities']['vision']['avg_frames']/10.0  # in seconds
        logging.info('duration: {} sec.'.format(self.duration))
        
        self.modalities_info = {}
        for modality in modalities_fps:
            self.modalities_info[modality] = {}
            if modality == 'text':
                continue
            elif modality == 'audio':
                self.modalities_info[modality]['shape'] = [self.metadata[self.behavior]['modalities'][modality]['avg_frames']]
            else:
                meta_mod = modality
                if modality in ['image', 'video']:
                    meta_mod = 'vision'
                self.modalities_info[modality]['shape'] = list(self.metadata[self.behavior]['modalities'][meta_mod]['shape'])
                fps = modalities_fps[modality]
                if fps:
                    freq = int(fps * self.duration)
                    self.modalities_info[modality]['shape'].insert(0, freq)
                else:
                    self.modalities_info[modality]['shape'].insert(0, self.metadata[self.behavior]['modalities'][meta_mod]['avg_frames'])

            self.modalities_info[modality]['num_features'] = np.prod(self.modalities_info[modality]['shape'])
            self.modalities_info[modality]['dataset_shape'] = [len(self.data_filepaths)] + self.modalities_info[modality]['shape']
            
            logging.info('{}: {}'.format(modality, self.modalities_info[modality]))

    def __len__(self):
        return len(self.data_filepaths)

    def __getitem__(self, item):
        
        object_name = self.data_filepaths[item]['object_name']
        trial_num = self.data_filepaths[item]['trial_num']
        modalities_data = {'object_name': object_name, 'trial_num': trial_num}
        for modality in self.modalities_fps:
            if modality == 'text':
                text_dataset = self.objects_texts[object_name]
                modalities_data[modality] = text_dataset[np.random.choice(len(text_dataset))]
                continue
            bin_file = open(self.data_filepaths[item][modality], 'rb')
            data = pickle.load(bin_file)
            bin_file.close()

            if modality in ['image', 'video']:
                data = resize(data, self.modalities_info[modality]['shape'], preserve_range=True).astype('uint8')
            else:
                data = resize(data, self.modalities_info[modality]['shape'])

            modalities_data[modality] = data

        return modalities_data


class CY101DataModule(pl.LightningDataModule):
    def __init__(self, binary_dataset_path: str, metadata: dict, behavior: str, modalities_fps: dict,
                 batch_size: int = 32, num_workers: int = 0):
        super().__init__()

        self.binary_dataset_path = binary_dataset_path
        self.metadata = metadata
        self.behavior = behavior
        self.modalities_fps = modalities_fps
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = CY101Dataset(binary_dataset_path, metadata, behavior, modalities_fps)
        logging.info('dataset: {}, {}'.format(len(self.dataset), self.dataset.data_filepaths[:2]))
        modalities_data = self.dataset[10]
        for modality in modalities_data:
            if isinstance(modalities_data[modality], (np.ndarray, np.generic)):
                logging.info('{}: {}'.format(modality, modalities_data[modality].shape))
                logging.info('num_features: {}'.format(self.dataset.modalities_info[modality]['num_features']))
                logging.info('dataset_shape: {}'.format(self.dataset.modalities_info[modality]['dataset_shape']))
            else:
                logging.info('{}: {}'.format(modality, modalities_data[modality]))

    def prepare_fold(self, train_objects, test_objects):
        # Assign train/test datasets for use in dataloaders
        
        self.train_idx = []
        for train_object in train_objects:
            self.train_idx.extend(self.dataset.objects_trials_idx[train_object])
        np.random.shuffle(self.train_idx)
        
        self.test_idx = []
        for test_object in test_objects:
            self.test_idx.extend(self.dataset.objects_trials_idx[test_object])

    def train_dataloader(self):
        return DataLoader(self.dataset, sampler=self.train_idx, batch_size=self.batch_size,
                          num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.dataset, sampler=self.test_idx, batch_size=self.batch_size,
                          num_workers=self.num_workers) # Same as test_dataloader

    def test_dataloader(self):
        return DataLoader(self.dataset, sampler=self.test_idx, batch_size=self.batch_size,
                          num_workers=self.num_workers)


class CY101DatasetUnifyRepr(Dataset):
    def __init__(self, dataset_path, metadata, behavior):

        self.metadata = metadata
        self.behavior = behavior

        objects = self.metadata[behavior]['objects']
        trials = self.metadata[behavior]['trials']

        self.data_filepaths = []
        self.objects_trials_idx = {}
        i = 0
        for object_name in sorted(objects):
            if object_name in ['no_object']:
                continue
            for trial_num in sorted(trials):
                trial_data = {'object_name': object_name, 'trial_num': trial_num}                    
                trial_data_filepath = os.sep.join([dataset_path, object_name, trial_num + '.bin'])
                trial_data['unify_repr'] = trial_data_filepath
                self.data_filepaths.append(trial_data)

                self.objects_trials_idx.setdefault(object_name, [])
                self.objects_trials_idx[object_name].append(i)
                i += 1
        
        self.num_features = self[0]['unify_repr'].shape[0]
        self.dataset_shape = [len(self.data_filepaths)] + [self.num_features]

    def __len__(self):
        return len(self.data_filepaths)

    def __getitem__(self, item):
        
        object_name = self.data_filepaths[item]['object_name']
        trial_num = self.data_filepaths[item]['trial_num']
        data = {'object_name': object_name, 'trial_num': trial_num}
        
        bin_file = open(self.data_filepaths[item]['unify_repr'], 'rb')
        data['unify_repr'] = pickle.load(bin_file)
        bin_file.close()
        
        return data


class CY101DataModuleUnifyRepr(pl.LightningDataModule):
    def __init__(self, binary_dataset_path: str, metadata: dict, behavior: str, batch_size: int = 32, num_workers: int = 0):
        super().__init__()

        self.binary_dataset_path = binary_dataset_path
        self.metadata = metadata
        self.behavior = behavior
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = CY101DatasetUnifyRepr(binary_dataset_path, metadata, behavior)
        logging.info('dataset: {}, {}'.format(len(self.dataset), self.dataset.data_filepaths[:2]))
        data = self.dataset[10]
        for key in data:
            if isinstance(data[key], (np.ndarray, np.generic)):
                logging.info('num_features: {}'.format(self.dataset.num_features))
                logging.info('dataset_shape: {}'.format(self.dataset.dataset_shape))
            else:
                logging.info('{}: {}'.format(key, data[key]))

    def prepare_fold(self, train_objects, test_objects):
        # Assign train/test datasets for use in dataloaders
        
        self.train_idx = []
        for train_object in train_objects:
            self.train_idx.extend(self.dataset.objects_trials_idx[train_object])
        np.random.shuffle(self.train_idx)
        
        self.test_idx = []
        for test_object in test_objects:
            self.test_idx.extend(self.dataset.objects_trials_idx[test_object])

    def train_dataloader(self):
        return DataLoader(self.dataset, sampler=self.train_idx, batch_size=self.batch_size,
                          num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.dataset, sampler=self.test_idx, batch_size=self.batch_size,
                          num_workers=self.num_workers) # Same as test_dataloader

    def test_dataloader(self):
        return DataLoader(self.dataset, sampler=self.test_idx, batch_size=self.batch_size,
                          num_workers=self.num_workers)
