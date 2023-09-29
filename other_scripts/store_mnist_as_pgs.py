import os
import struct
import sys
import numpy

from array import array
from os import path
from PIL import Image #imported from pillow

from tqdm import tqdm


def read(dataset):
    print(os.listdir("raw"))
    if dataset is "training":
        fname_img = "raw/train-images-idx3-ubyte"
        fname_lbl = "raw/train-labels-idx1-ubyte"

    elif dataset is "testing":

        fname_img = "raw/t10k-images-idx3-ubyte"
        fname_lbl = "raw/t10k-labels-idx1-ubyte"

    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array("B", fimg.read())
    fimg.close()

    return lbl, img, size, rows, cols

# funtion to extract and  the MNIST dataset 
def write_dataset(labels, data, size, rows, cols, output_dir):
    os.makedirs(output_dir, exist_ok=True)


    # write data
    pbar = tqdm(enumerate(labels))
    for (i, label) in pbar:
        output_filename = path.join(output_dir, f"{i}_{label}.jpg")
        pbar.set_description("writing " + output_filename)

        with open(output_filename, "wb") as h:
            data_i = [
                data[ (i*rows*cols + j*cols) : (i*rows*cols + (j+1)*cols) ]
                for j in range(rows)
            ]
            data_array = numpy.asarray(data_i)


            im = Image.fromarray(data_array)
            im.save(output_filename)


if __name__ == "__main__":
    output_path = 'jpgs'

    for dataset in ["training", "testing"]:
        labels, data, size, rows, cols = read(dataset)
        write_dataset(labels, data, size, rows, cols,
                      path.join(output_path, dataset))