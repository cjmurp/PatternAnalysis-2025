import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import nibabel as nib
from PIL import Image  # Import PIL for image handling


class HipMRIProstateDataset(Dataset):
    """
    Dataset class for the HipMRI Prostate Dataset
    """

    def __init__(self, root_dir, transform=None):
        """
        Constructor for the HipMRI Prostate Dataset
        :param root_dir: Path to directory containing the HipMRI Prostate images
        :param transform: Optional transform to be applied on a sample. Default: transforms to Tensor
        """

        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([transforms.ToTensor()])

        # Grab all Nifti files
        self.file_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith('.nii.gz') or f.endswith('.nii')
        ]

        # Checks if self.file_paths is empty
        if len(self.file_paths) == 0:
            print("There are no .nii.gz files in this directory: {}".format(root_dir))

    def __len__(self):
        """
        Returns the number of samples in the dataset
        :return: (int) Length of the dataset
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset
        :param idx: (int) Desired index
        :return: (torch.Tensor) Transformed MRI data as a tensor at idx.
        """
        # Load Nifti image
        path = self.file_paths[idx]
        nifti_img = nib.load(path)
        img = nifti_img.get_fdata(caching='unchanged')

        # keep only the first slice if 3d
        if len(img.shape) == 3:
            img = img[:, :, 0]

        # Normalise data to[0, 1]
        img_data_normalized = (img - np.min(img)) / (np.max(img) - np.min(img))

        # Converts range to [0, 255]
        img_data_255 = (img_data_normalized * 255).astype(np.uint8)

        # Converts NumPy array to a PIL Image
        img = Image.fromarray(img_data_255)

        if self.transform:
            img = self.transform(img)

        return img

if __name__ == "__main__":

    path = "./../../../keras_slices_data/keras_slices_validate"

    data = HipMRIProstateDataset(root_dir=path, transform=transforms.ToTensor())

    dataloader = DataLoader(data, batch_size=1, shuffle=True)
    for batch in dataloader:
        print(batch.shape)
        print(len(dataloader))
        break