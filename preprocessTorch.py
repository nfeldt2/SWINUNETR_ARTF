from skimage.transform import resize
import torch
import os
import numpy as np
import argparse

def generate_3d_torch(directory, data_name, LR=False):
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isdir(full_path):
            print('Directory:', full_path)
            generate_3d_torch(full_path, data_name, LR)
        else:
            if 'test' in directory:
                pass
            elif 'validate' in directory:
                pass
            elif 'train' in directory:
                pass
            else:
                continue
            if item.endswith('.npy'):
                data = np.load(full_path)
                if 'maskArtifact_' in item:
                    folder = 'labelsTr'
                    ending = '.pt'
                elif 'ROI' in item:
                    folder = 'roiTr'
                    ending = '.pt'
                else:
                    folder = 'imagesTr'
                    ending = '_0000.pt'

                if 'test' in directory:
                    folder = folder.replace('Tr', 'Ts')

                item = file_rename(item)

                if not os.path.isdir(f"/raid/{data_name}/{folder}"):
                    if not os.path.isdir(f"/raid/{data_name}"):
                        os.mkdir(f"/raid/{data_name}")
                    os.mkdir(f"/raid/{data_name}/{folder}")

                if LR:
                    if (("_L_" in item) or ("_R_" in item)) and ("roi" in folder):

                        data = torch.tensor(data)
                        torch.save(data, f"/raid/{data_name}/{folder}/{item[:-4]}{ending}")
                    if (not "roi" in folder):
                        data = torch.tensor(data)
                        torch.save(data, f"/raid/{data_name}/{folder}/{item[:-4]}{ending}")
                else:
                    if ("_L_" in item) or ("_R_" in item):
                        continue
                    data = torch.tensor(data)
                    torch.save(data, f"/raid/{data_name}/{folder}/{item[:-4]}{ending}")

def file_rename(name):
    if 'CTinhaleArtf_' in name:
        name = name.replace('CTinhaleArtf_', '')
    if 'maskArtifactROI_' in name:
        name = name.replace('maskArtifactROI_', '')
    else:
        name = name.replace('maskArtifact_', '')

    if 'All_' in name:
        name = name.replace('All_', '')
    if 'All(2023-07-13)_' in name:
        name = name.replace('All(2023-07-13)_', '')
    return name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--name", type=str, default="bingus")
    parser.add_argument("--LR", type=bool, default=False)
    args = parser.parse_args()
    directory = args.dir
    data_name = args.name
    LR = args.LR
    temp_name = directory.split('/')[-1]
    if LR:
        data_name += "_LR"
    data_name = f"Torch_{data_name}"
    generate_3d_torch(directory, data_name, LR)