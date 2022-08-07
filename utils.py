import os, shutil

TOTAL_SHARD = 4

for i in range(1, TOTAL_SHARD+1):
    shutil.copytree(os.path.join(str(i), 'tfrec'), os.path.join('train', str(i)))
    shutil.copytree(os.path.join(str(i), 'val_tfrec'), os.path.join('val', str(i)))
