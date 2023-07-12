import os

import numpy as np
from PIL import Image
from tqdm import tqdm

# root = '/cs/labs/yweiss/ariel1/data'
root = '/mnt/storage_ssd/datasets/'

FFHQ_1000 = f'{root}/FFHQ/FFHQ_1000/FFHQ_1000'

out_path = f'{root}/FFHQ/FFHQ64_1000_shuffled_asdasdsadasd/FFHQ64_1000_shuffled'
os.makedirs(out_path, exist_ok=True)

perm = None
for fname in tqdm(os.listdir(FFHQ_1000)):
    fpath = os.path.join(FFHQ_1000, fname)
    img_org = np.array(Image.open(fpath).resize((64,64)))
    img = img_org.reshape(-1, img_org.shape[-1])

    if perm is None:
        perm = np.random.permutation(img.shape[0])
        np.save(f'{os.path.dirname(out_path)}/perm.npy', perm)

    img = img[perm]
    img = img.reshape(img_org.shape)

    Image.fromarray(img).save(os.path.join(out_path, fname))

