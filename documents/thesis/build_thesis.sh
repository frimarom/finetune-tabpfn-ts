#!/bin/bash

set -e

OUT=out
mkdir -p "$OUT"

pdflatex -output-directory="$OUT" thesis.tex
bibtex "$OUT"/thesis
pdflatex -output-directory="$OUT" thesis.tex
pdflatex -output-directory="$OUT" thesis.tex
pdflatex -output-directory="$OUT" thesis.tex