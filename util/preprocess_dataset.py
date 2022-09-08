import json
import glob
import os
import gzip


TO_DECOMPRESS = ["hepaticvessel", "hippocampus", "pancreas", "prostate"]


def decompress(infile, tofile):
    with open(infile, 'rb') as inf, open(tofile, 'w', encoding='utf8') as tof:
        decom_str = gzip.decompress(inf.read()).decode('utf-8')
        tof.write(decom_str)


def process(config_path):
    config = json.load(open(config_path))

    for path in glob.glob(os.path.join(config['pretrain_data_path'], "*")):
        if os.path.basename(path).lower() not in TO_DECOMPRESS:
            continue

        # decompress training
        for prod_path in glob.glob(os.path.join(path, "training", "images", "*.gz")):
            os.system(f'gunzip {prod_path}')
        for prod_path in glob.glob(os.path.join(path, "training", "labels", "*.gz")):
            os.system(f'gunzip {prod_path}')

        # decompress testing
        for prod_path in glob.glob(os.path.join(path, "testing", "images", "*.gz")):
            os.system(f'gunzip {prod_path}')
        for prod_path in glob.glob(os.path.join(path, "testing", "labels", "*.gz")):
            os.system(f'gunzip {prod_path}')


if __name__ == '__main__':
    process('../config.json')
