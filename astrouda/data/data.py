from typing import Collection

import numpy as np
import os
import h5py
import urllib.request  

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from astrouda.utils import utils

class Download():
    def __init__(self, config):
        self.config = config
        self.logger = utils.make_logger(config)

    def download(self, dataset_paths: Collection[str]) -> None:

        def show_progress(block_num, block_size, total_size):
            self.logger.info(round(block_num * block_size / total_size *100,2), end="\r")

        # curl in training and validation data
        for url in dataset_paths:
            fp = os.path.join(self.config.get('source_dir', './data'), url.split("/")[-1].split("?")[0])
            if not os.path.exists(fp):
                self.logger.info((f"Downloading {url} to {fp}..."))
                urllib.request.urlretrieve(url, fp, reporthook=show_progress)
            else: 
                self.logger.info(f"File {fp} already exists, skipping download.")


    def lsst_y1(self):
        img_train = "https://zenodo.org/records/5514180/files/images_Y1_train.npy?download=1"
        label_train = "https://zenodo.org/records/5514180/files/labels_train.npy?download=1"

        img_val = "https://zenodo.org/records/5514180/files/images_Y1_valid.npy?download=1"
        label_val = "https://zenodo.org/records/5514180/files/labels_valid.npy?download=1"

        return [img_train, label_train, img_val, label_val]

    def lsst_y10(self):
        img_train = "https://zenodo.org/records/5514180/files/images_Y10_train.npy?download=1"
        label_train = "https://zenodo.org/records/5514180/files/labels_train.npy?download=1"

        img_val = "https://zenodo.org/records/5514180/files/images_Y10_valid.npy?download=1"
        label_val = "https://zenodo.org/records/5514180/files/labels_valid.npy?download=1"

        return [img_train, label_train, img_val, label_val]
    
    def decals(self):
        return ["https://zenodo.org/records/7473597/files/decals.h5?download=1"]

    def sdss_wide(self):
        return ["https://zenodo.org/records/7473597/files/sdss_1.h5?download=1", "https://zenodo.org/records/7473597/files/sdss_2.h5?download=1"]

    def sdss_stripe82(self):
        return ['https://zenodo.org/records/7473597/files/sdss_stripe82.h5?download=1']

    def __call__(self,):
        # If the data is not already present in source_dir, download it
        source_dataset = self.config.get("source_dataset", "LSST Y1")
        target_dataset = self.config.get("target_dataset", "LSST Y10")
        datasets = {
            "LSST Y1": self.lsst_y1,
            "LSST Y10": self.lsst_y10,
            "DeCals": self.decals,
            "SDSS Wide": self.sdss_wide,
            "SDSS Stripe82": self.sdss_stripe82
        }

        download_urls = set(datasets[source_dataset]()) | set(datasets[target_dataset]())
        self.download(download_urls) # only download files that are not already present


class DADataLoader(): 
    def __init__(self, config):
        self.config = config
        self.batch_size = config.get("batch_size", 32)
        self.shuffle = config.get("shuffle", True)

    def get_data_loader(self, train=True):
        dataset = DADataset(self.config, train)
        data_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle
        )
        return data_loader
    
class DADataset(Dataset):
    def __init__(self, config, train=True):
        self.config = config
        Download(config)() # download data if not already present
        
        self.train = train
        self.source_data, self.source_labels = self.load_data(self.config.get("source_dataset", "LSST Y1"))
        self.target_data, _ = self.load_data(self.config.get("target_dataset", "LSST Y10"))
        self.transform = self.preprocess_operations()

        if self.config.get("test", False):
            self.load_test_data()

    def load_lsst(self, source_dir, y1: bool=True) -> tuple[np.ndarray, np.ndarray]:
        name = 'Y1' if y1 else 'Y10'
        
        data = np.load(f"{source_dir.rstrip('/')}/images_{name}_{'train' if self.train else 'valid'}.npy") # (n_samples, 3, 100, 100)
        data = np.transpose(data, (0, 2, 3, 1)) # reshape to (n_samples, 100, 100, 3) for torchvision
        labels = np.load(f"{source_dir.rstrip('/')}/labels_train.npy")

        return data, labels

    def load_decals(self, source_dir) -> tuple[np.ndarray, np.ndarray]:
        # Download just zips it
        with h5py.File(f"{source_dir.rstrip('/')}/decals.h5", "r") as f:
            data = f["images"][:] # (n_samples, 100, 100, 3)
            labels = f["labels"][:]
        return data, labels


    def load_data(self, dataset_name) -> tuple[np.ndarray, np.ndarray|None]:
        """Load one of the datasets, return data and labels (if they exist)"""

        source_dir = self.config.get("source_dir", "./data")
        # LSST Mock
        # Sourced from DeepAdversaries - https://zenodo.org/records/5514180#.Y6SM7y-B2_w
        if dataset_name == "LSST Y1":
            data, labels = self.load_lsst(source_dir, y1=True)

        elif dataset_name == "LSST Y10":
            data, labels = self.load_lsst(source_dir, y1=False)

        # for SSD & DeCals
        elif dataset_name == "DeCals":
            data, labels = self.load_decals(source_dir)

        elif dataset_name in ["SDSS Wide", "SDSS Stripe82"]:
            raise NotImplementedError(f"Dataset {dataset_name} not yet implemented.")


        return data, labels

    def _test_sources(self, dataset_name) -> tuple[np.ndarray, np.ndarray|None]:
        def load_lsst_test(source_dir, y1: bool=True) -> tuple[np.ndarray, np.ndarray]:
            name = 'Y1' if y1 else 'Y10'
            data = np.load(f"{source_dir.rstrip('/')}/images_{name}_test.npy") # (n_samples, 3, 100, 100)
            data = np.transpose(data, (0, 2, 3, 1)) # reshape to (n_samples, 100, 100, 3) for torchvision
            labels = np.load(f"{source_dir.rstrip('/')}/labels_test.npy")
            return data, labels
        
        def load_decals_test(source_dir) -> tuple[np.ndarray, np.ndarray]:
            with h5py.File(f"{source_dir.rstrip('/')}/decals.h5", "r") as f:
                data = f["images_test"][:] # (n_samples, 100, 100, 3)
                labels = f["labels_test"][:]
            return data, labels
        
        def load_sdss_test(source_dir) -> tuple[np.ndarray, np.ndarray]:
            with h5py.File(f"{source_dir.rstrip('/')}/sdss_stripe82.h5", "r") as f:
                data = f["images_test"][:] # (n_samples, 100, 100, 3)
                labels = f["labels_test"][:]
            return data, labels
        
        def load_sdss_wide_test(source_dir) -> tuple[np.ndarray, np.ndarray]:
            with h5py.File(f"{source_dir.rstrip('/')}/sdss_1.h5", "r") as f:
                data1 = f["images_test"][:] # (n_samples, 100, 100, 3)
                labels1 = f["labels_test"][:]
            with h5py.File(f"{source_dir.rstrip('/')}/sdss_2.h5", "r") as f:
                data2 = f["images_test"][:] # (n_samples, 100, 100, 3)
                labels2 = f["labels_test"][:]
            data = np.concatenate([data1, data2], axis=0)
            labels = np.concatenate([labels1, labels2], axis=0)
            return data, labels
        
        source_dir = self.config.get("source_dir", "./data")
        test_data = {
            "LSST Y1": lambda: load_lsst_test(source_dir, y1=True),
            "LSST Y10": lambda: load_lsst_test(source_dir, y1=False),
            "DeCals": lambda: load_decals_test(source_dir),
            "SDSS Wide": lambda: load_sdss_wide_test(source_dir),
            "SDSS Stripe82": lambda: load_sdss_test(source_dir)
        }
        return test_data[dataset_name]()
    

    def load_test_data(self):
        source = self.config.get("test_dataset_source") 
        target = self.config.get("test_dataset_target")

        self.source_data, self.source_labels = self._test_sources(source)
        self.target_data, _ = self._test_sources(target)


    def preprocess_operations(self):
        transformations = [                         
            transforms.ToTensor(),
            transforms.Resize(self.config.get("resize_size", 256)),
            transforms.RandomCrop(self.config.get("crop_size", 224)),
            transforms.Normalize(0.5, 0.5)
        ]
        return transforms.Compose(transformations)

    def __len__(self):
        return min(len(self.source_data), len(self.target_data))
    
    def __getitem__(self, idx):
        source_image = self.transform(self.source_data[idx]).float()
        if self.source_labels is not None:
            source_label = self.source_labels[idx]
        else:
            source_label = None
        target_image = self.transform(self.target_data[idx]).float()
        return source_image, source_label, target_image
