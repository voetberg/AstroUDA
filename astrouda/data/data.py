import numpy as np

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

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
        self.train = train
        self.source_data, self.source_labels = self.load_data(self.config.get("source_dataset", "LSST Y1"))
        self.target_data, _ = self.load_data(self.config.get("target_dataset", "LSST Y10"))
        self.transform = self.preprocess_operations()

    def load_data(self, dataset_name) -> tuple[np.ndarray, np.ndarray|None]:
        """Load one of the datasets, return data and labels (if they exist)"""
        source_dir = self.config.get("source_dir", "./data")
        # LSST Mock
        # Sourced from DeepAdversaries - https://zenodo.org/records/5514180#.Y6SM7y-B2_w
        if dataset_name == "LSST Y1":
            data = np.load(f"{source_dir.rstrip('/')}/images_Y1_test_150.npy") # original shape - (n_samples, 3, 100, 100)
            data = np.transpose(data, (0, 2, 3, 1)) # reshape to (n_samples, 100, 100, 3) for torchvision
            labels = np.load(f"{source_dir.rstrip('/')}/labels_test_150.npy")

        elif dataset_name == "LSST Y10":
            data = np.load(f"{source_dir.rstrip('/')}/images_Y10_test_150.npy")
            data = np.transpose(data, (0, 2, 3, 1))
            labels = np.load(f"{source_dir.rstrip('/')}/labels_test_150.npy")


        # GZ2 SDSS
        # GZ3 DECals
        
        # GZ2 SDSS WIDE
        # Stripe 82
        return data, labels


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
