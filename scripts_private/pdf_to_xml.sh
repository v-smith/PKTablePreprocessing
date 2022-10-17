#!/usr/bin/env sh
#Use GroBid Restful API https://komax.github.io/blog/text/mining/grobid/
python3 grobid-client.py --n 3 --input ~/papers  --output ~/tei_papers  processFulltextDocument
# change name of directory of papers and name of directory for output
