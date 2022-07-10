import os

import numpy as np


class Dataset(object):

    def __init__(self, data_path):
        self.data_path = data_path
        assert os.path.exists(self.data_path), f'[ERROR] {data_path} is not a valid path!'

        img_suffixes = ('.png', '.PNG', '.jpeg', '.JPEG', '.jpg', '.JPG')

        self.labeled = True

        if np.all([f.endswith(img_suffixes) for f in os.listdir(data_path)]):
            print(f'[INFO] Does the specified folder contains class subdirectories? Setting labeled=True')
            self.labeled = False

        if self.labeled:
            self.data_classes = os.listdir(self.data_path)
            self.data_mapping = dict()

            for c, c_name in enumerate(self.data_classes):
                temp_path = os.path.join(self.data_path, c_name)
                temp_images = os.listdir(temp_path)

                for i in temp_images:
                    img_tmp = os.path.join(temp_path, i)

                    if img_tmp.lower().endswith(('.jpg', '.jpeg')):
                        if c_name == 'distractor':
                            self.data_mapping[img_tmp] = -1
                        else:
                            self.data_mapping[img_tmp] = c_name

            print(f'[INFO] Loaded {len(self.data_mapping.keys())} from {self.data_path} images')

        else:
            print(f'[INFO] Loaded {len(os.listdir(self.data_path))} from {self.data_path} images')

    def get_data_paths(self):
        '''
        Returns a list of img paths and related classes
        '''
        if self.labeled:
            images = []
            for img_path in self.data_mapping.keys():
                if img_path.lower().endswith(('.jpg', '.jpeg')):
                    images.append(img_path)
            return images
        else:
            data_paths = [os.path.join(self.data_path, img_path) for img_path in os.listdir(self.data_path)]
            return data_paths

    def get_data_classes(self):
        '''
        Returns a list of img paths and related classes
        '''
        if self.labeled:
            classes = []
            for img_path in self.data_mapping.keys():
                if img_path.lower().endswith(('.jpg', '.jpeg')):
                    classes.append(self.data_mapping[img_path])
            return np.array(classes)
        else:
            print('[WARN] The given dataset is unlabeled, no number of classes returned!')

    def num_classes(self):
        '''
        Returns number of classes of the dataset (if labeled)
        '''
        if not self.labeled:
            return len(self.data_classes)
        else:
            print('[WARN] The given dataset is unlabeled, no number of classes returned!')
