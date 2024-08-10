import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CMDataset(Dataset):
    def __init__(self, images_dir, confidence_maps_dir, split="train"):
        # Initialize dataset, you might need to read file paths and labels
        self.images_dir = images_dir
        self.confidence_maps_dir = confidence_maps_dir

        self.image_paths = []
        self.cm_paths = []

        self.val_participants = ["ABO_SH", "AUR_GR", "BOR_MA", "BOU_RO", "CHI_MO"]

        self.test_participants = [
            "DAB_AN",
            "DEG_VI",
            "DOG_VA",
            "GAB_TH",
            "GOY_MA",
            "LAB_AL",
            "MEN_GW",
            "REV_PA",
            "TRI_AN",
            "WAN_IK",
        ]

        participants = None

        if split == "train":
            participants = [
                participant
                for participant in os.listdir(images_dir)
                if participant not in [self.val_participants + self.test_participants]
            ]

        elif split == "val":
            participants = [
                participant
                for participant in os.listdir(images_dir)
                if participant in self.val_participants
            ]

        elif split == "test":
            participants = [
                participant
                for participant in os.listdir(images_dir)
                if participant in self.test_participants
            ]

        else:
            raise

        for participant in participants:
            participant_images_dir = os.path.join(images_dir, participant)
            participant_conf_maps_dir = os.path.join(confidence_maps_dir, participant)

            for slice_name in os.listdir(participant_images_dir):
                image_path = os.path.join(participant_images_dir, slice_name)
                cm_path = os.path.join(participant_conf_maps_dir, slice_name)

                self.image_paths.append(image_path)
                self.cm_paths.append(cm_path)

    def __len__(self):
        return len(self.image_paths)

    def transform(self, image, conf_map):
        transform_pipeline = transforms.Compose(
            [
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        image = transform_pipeline(image)
        conf_map = transform_pipeline(conf_map)

        return image, conf_map

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        conf_map_path = self.cm_paths[idx]

        image = Image.open(image_path).convert("L")
        conf_map = Image.open(conf_map_path).convert("L")

        image, conf_map = self.transform(image, conf_map)

        image = image.float()
        conf_map = conf_map.float()

        return image, conf_map
