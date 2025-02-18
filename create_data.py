import argparse
import h5py
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path

def fft(img):
    kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
    return kspace

def ifft(kspace):
    img = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(kspace, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
    return img

def center_crop(img, shape):
    """
    Center crops a 2D image to the specified shape.
    
    Args:
        img (np.ndarray): Input image.
        shape (tuple): Shape of the cropped image.
        
    Returns:
        np.ndarray: Cropped image.
    """
    _, h, w = img.shape
    new_h, new_w = shape
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return img[:, top:top + new_h, left:left + new_w]
    
    
def extract_random_slices(input_dir, output_dir, num_slices=5, crop_shape=(320, 320)):
    """
    Extracts random slices from 3D k-space data stored in .h5 files and saves them as .npz files.
    
    Args:
        input_dir (str or Path): Path to the directory containing .h5 files.
        output_dir (str or Path): Path to the directory where .npz files will be saved.
        num_slices (int): Number of random slices to extract from each file.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    h5_files = list(input_dir.glob("*.h5"))
    
    for file in tqdm(h5_files, desc="Processing files"):
        with h5py.File(file, 'r') as h5f:
            if 'kspace' not in h5f:
                print(f"Skipping {file} (no 'kspace' key found)")
                continue
            
            kspace = h5f['kspace'][:]
            num_available_slices = kspace.shape[0]  # Assuming shape (slices, channels, height, width)
            
            if num_available_slices < num_slices:
                print(f"Skipping {file} (not enough slices, found {num_available_slices})")
                continue
            
            selected_slices = random.sample(range(num_available_slices), num_slices)
            
            for i, slice_idx in enumerate(selected_slices):
                slice_data = kspace[slice_idx]
                
                # crop the center of the image
                slice_image = ifft(slice_data)
                slice_image_cropped = center_crop(slice_image, crop_shape)
                slice_data = fft(slice_image_cropped)
                
                output_filename = f"{file.stem}_slice{slice_idx}.npz"
                output_path = output_dir / output_filename
                
                np.savez_compressed(output_path, fullysampled_kspace=slice_data)
                
                print(f"Saved slice {i+1}/{num_slices} from {file} to {output_path} with shape {slice_data.shape}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Extract random slices from 3D k-space data stored in .h5 files.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--num_slices", type=int, default=5)
    
    args = parser.parse_args()
  
    input_directory = Path.home() / "storage/datasets/FastMRI/knee/multi_coil" / Path(args.mode)
    output_directory = Path.home() / "storage/datasets/FastMRI/knee/multi_coil" / Path(args.mode + "_2Dslices")
    
    # check if the input directory exists
    if not input_directory.exists():
        raise FileNotFoundError(f"Directory not found: {input_directory}")
    print(f"Extracting random slices from {input_directory} to {output_directory}...")
    
    extract_random_slices(input_directory, output_directory, num_slices=args.num_slices)