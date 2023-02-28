#! /usr/bin/python


import sys
import glob
import h5py


files = glob.glob(sys.argv[1])

writer = h5py.File('train.h5', 'w')

for i, f in enumerate(files):
    reader = h5py.File(f, 'r')
    for key in list(reader.keys()):
        writer.create_dataset("%s_%s" % (key, i), data=reader[key][()])
    reader.close()

writer.close()
