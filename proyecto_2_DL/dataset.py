""" 
This file is used to load the FER2013 dataset.
It consists of 48x48 pixel grayscale images of faces 
with 7 emotions - angry, disgust, fear, happy, sad, surprise, and neutral.
"""

import pathlib
from typing import Any, Callable, Optional, Tuple
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import cv2
import os
import numpy as np
from utils import to_numpy, to_torch, add_img_text, get_transforms
import json

EMOTIONS_MAP = {
    0: "Enojo",
    1: "Disgusto",
    2: "Miedo",
    3: "Alegria",
    4: "Tristeza",
    5: "Sorpresa",
    6: "Neutral"
}
file_path = pathlib.Path(__file__).parent.absolute()

def get_loader(split, batch_size, shuffle=True, num_workers=0):
    '''
    Get train and validation loaders
    args:
        - batch_size (int): batch size
        - split (str): split to load (train, test or val)
    '''
    dataset = FER2013(root=file_path,
                      split=split)
    dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
        )
    return dataset, dataloader

class FER2013(Dataset):
    """`FER2013
    <https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``root/fer2013`` exists.
        split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.img_size = 48
        self.target_transform = target_transform
        self.split = split
        self.root = root
        self.unnormalize = None
        self.transform, self.unnormalize = get_transforms(
            split=self.split,
            img_size=self.img_size
        )

        df = self._read_data()
        _str_to_array = [np.fromstring(val,  dtype=int, sep=' ')
                         for val in df['pixels'].values]
        
        self._samples = np.array(_str_to_array)
        if split == "test":
            self._labels = np.empty(shape=len(self._samples))
        else:
            self._labels = df['emotion'].values

    def _read_data(self):
        base_folder = pathlib.Path(self.root) / "data"
        
        _split = "train" if self.split == "train" or "val" else "test"
        file_name = f"{_split}.csv"
        data_file = base_folder / file_name

        if not os.path.isfile(data_file.as_posix()):
            raise RuntimeError(
                f"{file_name} not found in {base_folder} or corrupted. "
                f"You can download it from "
                f"https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge"
            )

        df = pd.read_csv(data_file)
        if self.split != "test":
            train_val_split = json.load(open(base_folder / "split.json", "r"))
            split_samples = train_val_split[self.split]
            df = df.iloc[split_samples]
        return df
        
    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        _vector_img = self._samples[idx]

        # Pre procesamiento de la imagen
        sample_image = _vector_img.reshape(self.img_size,
                                           self.img_size).astype('uint8')
        if self.transform is not None:
            image = self.transform(sample_image) # float32
        else:
            image = torch.from_numpy(sample_image) # uint8

        # Pre procesamiento de la etiqueta
        target = self._labels[idx]
        emotion = EMOTIONS_MAP[target]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"transformed": image,
                "label": target,
                "original": sample_image,
                "emotion": emotion}

def main():
    # Visualizar de una en una imagen
    split = "train"
    dataset, dataloader = get_loader(split=split, batch_size=1, shuffle=False)
    print(f"Loading {split} set with {len(dataloader)} samples")
    for datapoint in dataloader:
        transformed = datapoint['transformed']
        original = datapoint['original']
        label = datapoint['label']
        emotion = datapoint['emotion'][0]

        # Si se aplico alguna normalizacion, deshacerla para visualizacion
        if dataset.unnormalize is not None:
            # Espera un tensor
            transformed = dataset.unnormalize(transformed)

        # Transformar a numpy
        original = to_numpy(original)  # 0 - 255
        transformed = to_numpy(transformed)  # 0 - 1
        # transformed = (transformed * 255).astype('uint8')  # 0 - 255

        # Aumentar el tamaño de la imagen para visualizarla mejor
        viz_size = (200, 200)
        original = cv2.resize(original, viz_size)
        transformed = cv2.resize(transformed, viz_size)

        # Concatenar las imagenes, tienen que ser del mismo tipo
        original = original.astype('float32') / 255
        np_img = np.concatenate((original,
                                 transformed), axis=1)

        np_img = add_img_text(np_img, emotion)

        cv2.imshow("img", np_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()