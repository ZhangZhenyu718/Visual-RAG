#!/bin/bash
# Build the thesis PDF from the markdown chapters. Single standalone pandoc call so
# pandoc's own (complete, correct) LaTeX preamble handles tables/fonts/math. Chapters
# stay modular as ../<chapter>.md; metadata (title/abstract/class) is in meta.yaml.
# To use the school's template later: pandoc ... --template=school_template.tex
set -e
cd "$(dirname "$0")"

CHAPTERS="ch1_introduction ch2_related_work ch3_system_design ch4_implementation \
ch5_evaluation ch6_discussion ch7_conclusion"

tmp="$(mktemp -t thesisXXXX).md"
preprocess() {
  # drop HTML comment blocks; strip "Chapter N."/"Appendix A." from # headings and
  # "N.M "/"A.1 " prefixes from ##/### headings (LaTeX auto-numbers both).
  sed -e '/<!--/,/-->/d' "$1" \
    | sed -E 's/^# (Chapter [0-9]+\.|Appendix [A-Z]\.) /# /' \
    | sed -E 's/^(#{2,3}) ([A-Z]\.)?[0-9]+(\.[0-9]+)* /\1 /' \
    | sed -e 's/≈/approximately /g' -e 's/✓/yes/g' \
          -e 's/10⁴/10,000/g' -e 's/∧/and/g' -e 's/∈/in/g'
}

for c in $CHAPTERS; do preprocess "../chapters/$c.md" >> "$tmp"; printf '\n\n' >> "$tmp"; done
# References chapter (citeproc fills the #refs div; unnumbered, placed before the appendix):
printf '\n\n# References {.unnumbered}\n\n::: {#refs}\n:::\n\n' >> "$tmp"
printf '\n\n```{=latex}\n\\appendix\n```\n\n' >> "$tmp"
preprocess "../chapters/appendix_b_tables.md" >> "$tmp"

pandoc "$tmp" --metadata-file=meta.yaml -H header.tex \
  --top-level-division=chapter --number-sections --toc --toc-depth=1 \
  --citeproc --csl=numeric.csl --bibliography=../chapters/references.bib \
  --pdf-engine=xelatex -o main.pdf > build.log 2>&1 || {
    echo "pandoc/pdflatex FAILED — tail of build.log:"; tail -30 build.log; rm -f "$tmp"; exit 1; }
rm -f "$tmp"
pages=$(pdfinfo main.pdf 2>/dev/null | awk '/Pages/{print $2}')
echo "OK: built main.pdf (${pages:-?} pages)"
