#! /usr/bin/python

import sys
import os

csv = open(sys.argv[1], 'r')
csv.readline()

root = sys.argv[2]
name = sys.argv[3]

writer_img = open("%s.img" % name, 'w')
writer_src = open("%s.src" % name, 'w')
writer_tgt = open("%s.tgt" % name, 'w')

for sample in csv:
    segs = sample.strip().split('|')

    img = segs[0]
    src = segs[-2]
    tgt = segs[-1]

    writer_img.write(os.path.abspath(os.path.join(root, name, img)) + '\n')
    writer_src.write(src + '\n')
    writer_tgt.write(tgt + '\n')

writer_img.close()
writer_src.close()
writer_tgt.close()
