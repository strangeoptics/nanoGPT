import os
import argparse
import random
import tiktoken
import numpy as np

def humanbytes(B):
    """Return the given bytes as a human friendly KB, MB, GB, or TB string."""
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2) # 1,048,576
    GB = float(KB ** 3) # 1,073,741,824
    TB = float(KB ** 4) # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B,'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B / KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B / MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B / GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B / TB)

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, help='name of the source folder with text files', required=True)
parser.add_argument('--nbr', type=int, help='number of files from folder')
args = parser.parse_args()

print('\nreading content of folder: ', args.folder)

data = ""
filenames = os.listdir(args.folder)

if args.nbr is not None:
    random.shuffle(filenames)
    filenames = filenames[:args.nbr]

for filename in filenames:
    name = args.folder+'/'+filename
    print('   ', name)
    with open(name, 'r',  encoding="utf-8") as f:
        data += f.read()

n = len(data)        
print('\ndata : ', humanbytes(n))

train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]
print('train:  ' + str(humanbytes(len(train_data))))
print('val:    ' + str(humanbytes(len(val_data))))

# encode with tiktoken gpt2 bpe (byte pair encoding)
print('\nencode with tiktoken gpt2 bpe (byte pair encoding)')
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train : {len(train_ids)} tokens")
print(f"val   : {len(val_ids)} tokens")

# export to bin files
print('\nexport to bin files')
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile('train.bin')
val_ids.tofile('val.bin')