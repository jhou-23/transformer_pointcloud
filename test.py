import os
import json

fi = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '%s_files.json' %'test'), 'r')
fns = list(json.load(fi))
fi.close()

for fname in fns:
    p = os.path.join("modelnet40", fname)
    f = open(p, 'r')
    lines = []
    for line in f:
        lines.append(line)
    f.close()
    if lines[0] != "OFF\n":
        meta = lines[0][3:]
        lines[0] = "OFF\n"
        lines.insert(1, meta)
        f = open(p, 'w')
        for line in lines:
            f.write(line)
        f.close()