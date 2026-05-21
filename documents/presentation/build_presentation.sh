#!/bin/bash

set -e

OUT=out
mkdir -p "$OUT"

pdflatex -output-directory="$OUT" slides.tex
pdflatex -output-directory="$OUT" slides.tex