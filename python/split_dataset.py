import os
import json

DADA_path = r'D:\DADA_dataset'  # change to your own path
link_paths = [r'D:\DADA2000\train',
              r'D:\DADA2000\test',
              r'D:\DADA2000\val',
              ]  # change to your own path, and in the same order as the json file below.

json_files = [r"D:\train_file.json",
              r"D:\test_file.json",
              r"D:\val_file.json",
              ]  # change to your own path

if __name__ == '__main__':

    for json_file, link_path in zip(json_files, link_paths):
        with open(json_file, 'r') as f:
            train_f = json.load(f)
            for file in train_f:
                save_p = os.path.join(link_path, file[1])
                src = os.path.join(DADA_path, file[0][0], file[0][1])
                os.symlink(src, save_p, target_is_directory=True)
