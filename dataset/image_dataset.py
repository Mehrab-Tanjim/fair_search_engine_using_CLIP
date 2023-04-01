from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import torchvision

class ImageDataset(Dataset):
    def __init__(self, image_paths, label_path, preprocess) -> None:
        super().__init__()

        self.image_paths = np.array(image_paths)
        
        self.labels_map = {}
        self.labels = []

        with open(label_path, 'r') as f:
            next(f) #skip
            self.attributes = next(f).strip().split() # get the attributes
            for line in f:
                fields = line.strip().split()
                image_name = fields[0] # filename
                labels = [1 if field=='1' else 0 for field in fields[1:]] # attribute
                self.labels_map[image_name] = np.array(labels)
            
        assert len(self.image_paths) == len(self.labels_map.keys()) # sanity check
        self.total_num_images = len(self.image_paths)

        for image_path in self.image_paths:
            image_name = os.path.split(image_path)[-1].lower().replace('.png', '.jpg')
            self.labels.append(self.labels_map[image_name])
        
        self.labels = np.array(self.labels)
        if preprocess is not None:
            self.transform = preprocess
        else:
            self.transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()
                            ])

    def sort_image_paths(self, attribute):
        label_for_the_target_attribute = self.labels[:, self.attributes.index(attribute)]
        sorted_index = np.argsort(-label_for_the_target_attribute)
        self.image_paths = self.image_paths[sorted_index]


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image_name = os.path.split(image_path)[-1].lower().replace('.png', '.jpg')
        label = self.labels_map[image_name]
        image = self.transform(Image.open(self.image_paths[index]))

        image_dict = {'images': image, 'image_paths': image_path, 'labels': label}
        return image_dict
