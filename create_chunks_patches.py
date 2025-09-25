# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 22:47:15 2025

@author: advit
"""
import os
import rasterio 
import numpy as np

def create_chunked_numpy_arrays():
    DATA_DIR = r"E:\Disaster\new_data\photu"
    PATCH_SIZE = 256
    STRIDE = 256
    OUT_DIR = r"E:\patch_chunks"
    PATCHES_PER_CHUNK = 500
    
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".tif")])
    num_weeks = len(files)
    
    print(f"Found {num_weeks} .tif files")
    
    with rasterio.open(os.path.join(DATA_DIR, files[0])) as src:
        bands, H, W = src.count, src.height, src.width
        print(f"Image dimensions: {H}x{W}, Bands: {bands}")
        
        num_patches_h = (H - PATCH_SIZE) // STRIDE + 1
        num_patches_w = (W - PATCH_SIZE) // STRIDE + 1
        num_patches = num_patches_h * num_patches_w
        
        print(f"Will create {num_patches_h} x {num_patches_w} = {num_patches} patches")
        
        num_chunks = (num_patches + PATCHES_PER_CHUNK - 1) // PATCHES_PER_CHUNK
        
        metadata = {
            'patch_size': PATCH_SIZE,
            'stride': STRIDE,
            'bands': bands,
            'num_weeks': num_weeks,
            'total_patches': num_patches,
            'patches_per_chunk': PATCHES_PER_CHUNK,
            'num_chunks': num_chunks,
            'image_height': H,
            'image_width': W
        }
        np.save(os.path.join(OUT_DIR, 'metadata.npy'), metadata)
        print("Metadata saved successfully")
        
        patch_idx = 0
        chunk_idx = 0
        current_chunk_patches = []
        
        for i in range(0, H - PATCH_SIZE + 1, STRIDE):
            for j in range(0, W - PATCH_SIZE + 1, STRIDE):
                try:
                    patch_data = np.zeros((num_weeks, bands, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                    
                    # Read all weeks for this patch location
                    for week_idx, file_name in enumerate(files):
                        file_path = os.path.join(DATA_DIR, file_name)
                        
                        with rasterio.open(file_path) as src:
                            # Make sure we don't read beyond image bounds
                            if i + PATCH_SIZE <= H and j + PATCH_SIZE <= W:
                                window = rasterio.windows.Window(j, i, PATCH_SIZE, PATCH_SIZE)
                                patch = src.read(window=window).astype(np.float32)
                                
                                # Verify patch shape
                                if patch.shape == (bands, PATCH_SIZE, PATCH_SIZE):
                                    patch_data[week_idx] = patch
                                else:
                                    print(f"Warning: Unexpected patch shape {patch.shape} at position ({i}, {j})")
                            else:
                                print(f"Warning: Patch at ({i}, {j}) would exceed image bounds")
                    
                    current_chunk_patches.append(patch_data)
                    patch_idx += 1
                    
                    # Print progress every 50 patches
                    if patch_idx % 50 == 0:
                        print(f"Processed {patch_idx}/{num_patches} patches...")
                    
                    # Save chunk when it's full or we've processed all patches
                    if len(current_chunk_patches) == PATCHES_PER_CHUNK or patch_idx == num_patches:
                        chunk_array = np.array(current_chunk_patches, dtype=np.float32)
                        chunk_filename = os.path.join(OUT_DIR, f'chunk_{chunk_idx:03d}.npy')
                        np.save(chunk_filename, chunk_array)
                        
                        print(f"Saved chunk {chunk_idx} with {len(current_chunk_patches)} patches")
                        print(f"  Shape: {chunk_array.shape}")
                        print(f"  File: {chunk_filename}")
                        
                        # Reset for next chunk
                        current_chunk_patches = []
                        chunk_idx += 1
                        
                except Exception as e:
                    print(f"Error processing patch at ({i}, {j}): {e}")
                    continue
        
        print(f"Successfully created {chunk_idx} chunk files!")
        print(f"Total patches processed: {patch_idx}")

if __name__ == "__main__":
    create_chunked_numpy_arrays()