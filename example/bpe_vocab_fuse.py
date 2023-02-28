# coding: utf-8

import sys
import codecs
from collections import OrderedDict


codes = codecs.open(sys.argv[1], 'r', encoding='utf-8').readlines()
vocab = codecs.open(sys.argv[2], 'r', encoding='utf-8').readlines()


data = OrderedDict()

for v in vocab:
    data[v.strip()] = len(data)

charcters = []

first=True
for code in codes:
    if first:
        first = False
        continue

    a, b = code.strip().split()

    if '</w>' not in b:
        pair = '%s%s@@' % (a, b.replace('</w>', ''))
    else:
        pair = '%s%s' % (a, b.replace('</w>', ''))

    if pair not in data:
        data[pair] = len(data)

    for c in a+b.replace('</w>', ''):
        charcters.append(c)

chars = set(charcters)
for c in chars:
    if c not in data:
        data[c] = len(data)
        data[c+'@@'] = len(data)

writer = codecs.open(sys.argv[3], 'w', encoding='utf-8')
for v in data:
    writer.write(v + '\n')
writer.close()
