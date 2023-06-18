from scripts.EMD.dists import emd
from utils import get_data

if __name__ == '__main__':
    data_path = '/mnt/storage_ssd/datasets/FFHQ/FFHQ_128/FFHQ_128'
    b = 35000
    data = get_data(data_path, 64, 3, limit_data=2*b)

    r1 = data[:b]
    r2 = data[b:]

    loss = emd(r1, r2)

    print(loss)