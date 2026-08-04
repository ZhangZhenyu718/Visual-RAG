#!/bin/bash
# Build the CHINESE thesis PDF from ../zh/*.md via pandoc + xelatex (ctexrep class).
# Chinese chapter/section headings are written WITHOUT numbers in the md; ctexrep
# auto-numbers them ("第 X 章"). Figure paths and math are identical to the English build.
# Builds whatever zh/ chapters exist, so translation can proceed incrementally.
set -e
cd "$(dirname "$0")"

CHAPTERS="ch1_introduction ch2_related_work ch3_system_design ch4_implementation \
ch5_evaluation ch6_discussion ch7_conclusion"

tmp="$(mktemp -t thesiszhXXXX).md"
preprocess() {
  # strip single-line <!-- ... --> comments first, then any multi-line comment blocks
  sed -e 's/<!--.*-->//g' -e '/<!--/,/-->/d' "$1" \
    | sed -E 's/^# 第[一二三四五六七八九十]+章 /# /' \
    | sed -E 's/^(#{2,3}) [0-9]+(\.[0-9]+)+ /\1 /' \
    | sed -e 's/≈/约/g' -e 's/✓/是/g' \
          -e 's/10⁴/10,000/g' -e 's/∧/且/g' -e 's/∈/属于/g'
}

included=""
for c in $CHAPTERS; do
  if [ -f "../chapters/zh/$c.zh.md" ]; then
    preprocess "../chapters/zh/$c.zh.md" >> "$tmp"; printf '\n\n' >> "$tmp"; included="$included $c"
  fi
done
# References chapter (citeproc fills the #refs div; unnumbered, placed before the appendix):
printf '\n\n# 参考文献 {.unnumbered}\n\n```{=latex}\n\\markboth{参考文献}{参考文献}\n```\n\n::: {#refs}\n:::\n\n' >> "$tmp"
if [ -f "../zh/A_reproduction.md" ]; then
  printf '\n\n```{=latex}\n\\appendix\n```\n\n' >> "$tmp"
  preprocess "../zh/A_reproduction.md" >> "$tmp"; included="$included A"
fi
if [ -f "../zh/B_proof_details.md" ]; then
  printf '\n\n' >> "$tmp"
  preprocess "../zh/B_proof_details.md" >> "$tmp"; included="$included B"
fi
echo "chapters included:$included"

pandoc "$tmp" --metadata-file=meta_zh.yaml -H header.tex \
  --top-level-division=chapter --number-sections --toc --toc-depth=1 \
  --citeproc --csl=numeric.csl --bibliography=../chapters/references.bib \
  --pdf-engine=xelatex -o main_zh.pdf > build_zh.log 2>&1 || {
    echo "FAILED — tail of build_zh.log:"; tail -30 build_zh.log; rm -f "$tmp"; exit 1; }
rm -f "$tmp"
pages=$(pdfinfo main_zh.pdf 2>/dev/null | awk '/Pages/{print $2}')
echo "OK: built main_zh.pdf (${pages:-?} pages)"
