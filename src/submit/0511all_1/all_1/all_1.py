import io
import sys


infile = sys.argv[1]
outfile = sys.argv[2]

with io.open(infile, encoding="utf-8") as fr, io.open(outfile, "w", encoding="utf-8") as fw:
    for line in fr:
        id_ = line.split("\t")[0]
        fw.write(u"{}\t{}\n".format(id_, "1"))
