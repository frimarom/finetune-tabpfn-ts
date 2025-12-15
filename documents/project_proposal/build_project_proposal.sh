#!/bin/bash

set -e

#cd ./documents/project_proposal || exit

OUT=out
mkdir -p "$OUT"

pdflatex -output-directory="$OUT" project_proposal.tex
bibtex "$OUT"/project_proposal
pdflatex -output-directory="$OUT" project_proposal.tex
pdflatex -output-directory="$OUT" project_proposal.tex
