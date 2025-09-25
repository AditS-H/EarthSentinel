# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 23:05:55 2025

@author: advit
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import box
import random
import matplotlib.pyplot as plt
import os
import sys


if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

class Chunked_patch_loader:
    def __init__(self,chunk_dir):
        self.chunk_dir=chunk_dir
        print(f"Initialising chunked patch loader from : {chunk_dir}")
        metadata_path=os.path.join(chunk_dir,"metadata.npy")
        self.metadata=np.load(metadata_path,allow_pickle=True).item()
        print(f"Loaded metadata : {self.metadata}")
        
        self._chunk_cache={}
        self._cache_size_limit=3
        
    def get_patch_data(self,patch_indices,week_idx):
        patches_per_chunk=self.metadata['patches_per_chunk']
        patch_data=[]
        
        chunk_groups={}
        for patch_idx in patch_indices:
            chunk_idx=patch_idx//patches_per_chunk
            if chunk_idx not in chunk_groups:
                chunk_groups[chunk_idx]=[]
            chunk_groups[chunk_idx].append(patch_idx)
            
        for chunk_idx,patch_list in chunk_groups.items():
            chunk_data=self._load_chunk(chunk_idx)
            
            for patch_idx in patch_list:
                patch_in_chunk=patch_idx%patches_per_chunk
                if patch_in_chunk < chunk_data.shape[0]:
                    if week_idx is None:
                        patch_data.append(chunk_data[patch_in_chunk])
                    else:
                        patch_data.append(chunk_data[patch_in_chunk,week_idx])
                else:
                    print(f"patch {patch_idx} not found in chunk {chunk_idx}")
        return np.array(patch_data)
    
    def _load_chunk(self,chunk_idx):
        if chunk_idx in self._chunk_cache:
            return self._chunk_cache[chunk_idx]
        
        chunk_file=os.path.join(self.chunk_dir,f'chunk_{chunk_idx:03d}.npy')
        if not os.path.exists(chunk_file):
            raise FileNotFoundError(f"Chunk file not found {chunk_file}")
        
        chunk_data=np.load(chunk_file)
        print(f"Loaded chunk {chunk_idx}: shape {chunk_data.shape}")
        
        if len(self._chunk_cache) >= self._cache_size_limit:
            oldest_chunk=next(iter(self._chunk_cache))
            del self._chunk_cache[oldest_chunk]
            print(f"   [REMOVED] Removed chunk {oldest_chunk} from cache")
        
        self._chunk_cache[chunk_idx] = chunk_data
        return chunk_data
    
    def get_single_patch(self, patch_idx, week_idx):
        return self.get_patch_data([patch_idx], week_idx)[0]
    
    @property
    def shape(self):
        return (self.metadata['total_patches'], 
                self.metadata['num_weeks'], 
                self.metadata['bands'], 
                self.metadata['patch_size'], 
                self.metadata['patch_size'])
    
    
PATCH_SIZE=256
STRIDE=256
BANDS=4

H=15516
W=19020

CHUNK_DIR="patch_chunks"
GLC_CSV="Global_Landslide_Catalog_Export.csv"
HIMACHAL_GEOJSON="Himachal_GeoJSON.geojson"
RASTER_FOLDER="images/"
RASTER_FILES=[f"HP_week{i}_stack.tif" for i in range(1,9)]
WEEK_DATES = pd.to_datetime([
    '2016-06-01','2016-06-08','2016-06-15','2016-06-22', '2016-06-29','2016-07-06','2016-07-13','2016-07-20',
    '2016-07-27','2016-08-03','2016-08-10','2016-08-17',
    '2016-08-24','2016-08-31'
])

NUM_WEEKS = len(WEEK_DATES)
NUM_PAIRS_PER_WEEK = 1000

num_patches_h=(H-PATCH_SIZE)//STRIDE + 1
num_patches_w=(W-PATCH_SIZE)//STRIDE + 1
NUM_PATCHES=num_patches_h*num_patches_w

print("Loading chunked patches")
patch_loader=Chunked_patch_loader(CHUNK_DIR)

print("Chunked patch loader initialised")
print(f"Virtual shape : {patch_loader.shape}")

chunk_files = [f for f in os.listdir(CHUNK_DIR) if f.startswith('chunk_')]
total_size_gb = sum(os.path.getsize(os.path.join(CHUNK_DIR, f)) for f in chunk_files) / 1e9
print(f"   [FILES] Total chunk files size: {total_size_gb:.2f} GB")

print("Loading GLC CSV...")
df=pd.read_csv(GLC_CSV)
country_col='country' if 'country' in df.columns else 'country_name'
df_india=df[df[country_col].str.contains("India",case=False,na=False)].copy()
df_india['event_date'] = pd.to_datetime(df_india['event_date'], errors='coerce')

gdf_events = gpd.GeoDataFrame(
    df_india.dropna(subset=['latitude', 'longitude']),
    geometry=gpd.points_from_xy(df_india.longitude, df_india.latitude),
    crs="EPSG:4326"
)

himachal_shape = gpd.read_file(HIMACHAL_GEOJSON).to_crs(epsg=4326)
gdf_hp_events = gdf_events[gdf_events.within(himachal_shape.union_all())]
print(f" Himachal events : {len(gdf_hp_events)}")

raster_path = os.path.join(RASTER_FOLDER, RASTER_FILES[0])
with rasterio.open(raster_path) as src:
    transform = src.transform
    raster_crs = src.crs
    H_r, W_r = src.height, src.width

assert H == H_r and W == W_r, "Raster dimensions mismatch!"


gdf_hp_events_proj=gdf_hp_events.to_crs(raster_crs)

patch_labels=np.zeros((NUM_PATCHES,NUM_WEEKS),dtype=np.uint8)
patch_coords=[(i,j) for i in range(0,H-PATCH_SIZE+1,STRIDE)
                    for j in range(0,W-PATCH_SIZE+1,STRIDE)]

print("Analysing event-patch matching")
event_debug=[]
for w,window_end_date in enumerate(WEEK_DATES):
    print(f"Week {w} : {window_end_date.date()}")
    for idx,event in gdf_hp_events_proj.iterrows():
        if event['event_date']<=window_end_date:
            matched_patch=None
            for patch_idx,(i_start,j_start) in enumerate(patch_coords):
                ulx,uly=transform * (j_start,i_start)
                lrx,lry= transform * (j_start+PATCH_SIZE,i_start+PATCH_SIZE)
                patch_poly=box(ulx,lry,lrx,uly)
                if event.geometry.within(patch_poly):
                    matched_patch=patch_idx
                    break
            event_debug.append({
                'week_idx':w,
                'event_date':event['event_date'],
                'matched_patch':matched_patch,
                'status':'matched' if matched_patch is not None else 'unmatched'
                })
event_debug_df=pd.DataFrame(event_debug)
print("Event Patch debug :" )
print(event_debug_df.head(20))
summary = event_debug_df.groupby(['week_idx','status']).size().unstack(fill_value=0)
print("\n Week-wise matched/unmatched summary:")
print(summary)

print("Assigning temporal labels")
for patch_idx,(i_start,j_start) in enumerate(patch_coords):
    if patch_idx % 1000 == 0:
        print(f"   Processing patch {patch_idx}/{NUM_PATCHES}")
    ulx,uly=transform * (j_start,i_start)
    lrx,lry= transform * (j_start+PATCH_SIZE,i_start+PATCH_SIZE)
    patch_poly=box(ulx,lry,lrx,uly)
    for w, window_end_date in enumerate(WEEK_DATES):
        intersects = gdf_hp_events_proj[
            (gdf_hp_events_proj.geometry.within(patch_poly)) &
            (gdf_hp_events_proj.event_date <= window_end_date)
        ]
        patch_labels[patch_idx, w] = 1 if len(intersects) > 0 else 0

print("patch labels assigned . shape : ",patch_labels.shape)

all_pairs, all_pair_labels, all_weeks=[],[],[]

print("Generating  week-specific Siamese pairs...")
for w in range(NUM_WEEKS):
    pos_idx=np.where(patch_labels[:,w]==1)[0]
    neg_idx=np.where(patch_labels[:,w]==0)[0]
    print(f"   Week {w}: {len(pos_idx)} positive, {len(neg_idx)} negative patches")
    if len(pos_idx)<2 or len(neg_idx)<1:
        print(f"Week {w} : Not enough patches to generate siamese pairs.. skiping")
        continue
    pos_pairs = [(random.choice(pos_idx), random.choice(pos_idx))
                 for _ in range(NUM_PAIRS_PER_WEEK)]
    neg_pairs = [(random.choice(pos_idx), random.choice(neg_idx))
                 for _ in range(NUM_PAIRS_PER_WEEK)]
    pairs = pos_pairs + neg_pairs
    pair_labels = [1]*len(pos_pairs) + [0]*len(neg_pairs)
    week_ids = [w]*len(pairs)
    combined = list(zip(pairs, pair_labels, week_ids))
    random.shuffle(combined)
    pairs, pair_labels, week_ids = zip(*combined)
    all_pairs.extend(pairs)
    all_pair_labels.extend(pair_labels)
    all_weeks.extend(week_ids)

all_pairs = np.array(all_pairs)
all_pair_labels = np.array(all_pair_labels)
all_weeks = np.array(all_weeks)
print(" Week-specific Siamese pairs generated.")
print("Pairs shape:", all_pairs.shape)
print("Labels shape:", all_pair_labels.shape)
print("Week indices shape:", all_weeks.shape)

np.save("siamese_week_pairs.npy", all_pairs)
np.save("siamese_week_pair_labels.npy", all_pair_labels)
np.save("siamese_week_indices.npy", all_weeks)
print("[SUCCESS] Saved Siamese pairs, labels, and week indices.")


if len(all_pairs) > 0:
    print("[PREVIEW] Generating preview...")
    idx = random.randint(0, len(all_pairs)-1)
    a_idx, b_idx = all_pairs[idx]
    week = all_weeks[idx]
    
   
    patch_a = patch_loader.get_single_patch(a_idx, week)
    patch_b = patch_loader.get_single_patch(b_idx, week)
    
    num_bands = patch_a.shape[0]  
    
    plt.figure(figsize=(12, 6))
    
    for band in range(num_bands):
        # Patch A
        plt.subplot(2, num_bands, band+1)
        plt.imshow(patch_a[band], cmap="gray")
        plt.title(f"A idx={a_idx}, week={week}\nBand {band+1}, label={patch_labels[a_idx, week]}")
        plt.axis("off")

       
        plt.subplot(2, num_bands, num_bands + band + 1)
        plt.imshow(patch_b[band], cmap="gray")
        plt.title(f"B idx={b_idx}, week={week}\nBand {band+1}, label={patch_labels[b_idx, week]}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

print("[SUCCESS] Pipeline completed successfully with chunked data loading!")