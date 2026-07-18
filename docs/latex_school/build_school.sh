#!/bin/bash
# Build the University-of-Bristol-template thesis PDF.
# Pipeline: ../chapters/*.md --(preprocess+pandoc fragments)--> ch/*.tex,
# the abstract below --> abstract.tex, ../chapters/references.bib --> refs_clean.bib,
# then pdflatex + biber. Edit the markdown chapters, not ch/*.tex.
set -e
cd "$(dirname "$0")"

CHAPTER_SOURCES="ch1_introduction ch2_related_work ch3_system_design \
ch4_implementation ch5_evaluation ch6_discussion ch7_conclusion"

preprocess() {
  # same rules as latex/build.sh, plus raw "→" (pdflatex-safe via \ensuremath)
  sed -e '/<!--/,/-->/d' "$1" \
    | sed -E 's/^# Chapter [0-9]+ (—|-) /# /' \
    | sed -E 's/^# Appendix [A-Z]+ (—|-) /# /' \
    | sed -E 's/^(#{2,3}) ([A-Z]\.)?[0-9]+(\.[0-9]+)* /\1 /' \
    | sed -e 's/≈/approximately /g' -e 's/✓/yes/g' \
          -e 's/→/\\ensuremath{\\rightarrow}/g'
}

mkdir -p ch
GENERATED=""
chapter_no=1
for c in $CHAPTER_SOURCES; do
  out=$(printf '%02d_%s' "$chapter_no" "${c#ch?_}")
  preprocess "../chapters/$c.md" \
    | pandoc -f markdown -t latex --biblatex --syntax-highlighting=none \
        --top-level-division=chapter \
    | sed -e 's/\\def\\LTcaptype{none}//' -e 's/\\label{[^}]*}//g' \
    > "ch/$out.tex"
  GENERATED="$GENERATED ch/$out.tex"
  chapter_no=$((chapter_no + 1))
done
preprocess "../chapters/appendix_b_tables.md" \
  | pandoc -f markdown -t latex --biblatex --syntax-highlighting=none \
      --top-level-division=chapter \
  | sed -e 's/\\def\\LTcaptype{none}//' -e 's/\\label{[^}]*}//g' \
  > ch/A_full_results.tex
GENERATED="$GENERATED ch/A_full_results.tex"
# ^ pandoc marks caption-less longtables with \LTcaptype{none}; KOMA's longtable
#   support evaluates that as a counter name and errors ("No counter 'none'").
#   Our tables carry no captions, so dropping the marker changes nothing.
echo "fragments: $(printf '%s\n' $GENERATED | wc -l | tr -d ' ') chapters/appendices"

# Keep the project metadata here so this school-template build is self-contained.
python3 - <<'EOF'
import subprocess, pathlib
text = '''Video is widely recorded but remains difficult to search at the moment level, especially when a query expresses causal or temporal intent rather than literal visual content. This dissertation presents Visual RAG, an end-to-end agentic retrieval-augmented generation system that indexes timestamped visual and transcript evidence, decomposes natural-language questions into retrieval-friendly descriptions, searches and re-ranks candidate moments, and answers with grounded timestamp citations. The system is designed to run with modest resources by separating an offline indexing stage from a lightweight online query path.

The design is evaluated on 3,358 temporally grounded questions over 567 videos from NExT-QA and NExT-GQA. Visual retrieval substantially outperforms transcript retrieval at moment granularity. Replacing the baseline CLIP index with SigLIP and adding query decomposition nearly doubles corpus-scope recall at rank one from 0.026 to 0.048. A two-stage configuration using a cheaper index and a stronger visual re-ranker recovers most of the quality of an expensive index. Two negative results are also established: naive late fusion with sparse conversational transcripts and text cross-encoder re-ranking both reduce retrieval quality. At the answering stage, a bounded LangGraph agent improves five-choice accuracy from 0.447 to 0.547, while a controlled experiment on temporal questions shows that supplying visual evidence raises accuracy from 0.341 to 0.636.

The results separate three bottlenecks in video RAG: recall, ranking precision, and evidence delivery. Query decomposition, visual re-ranking, and multimodal answering address these bottlenecks independently and compose effectively. The dissertation contributes a reproducible system, evaluation harness, ablation study, and practical design guidance for temporally grounded video search on consumer hardware.'''
tex = subprocess.run(['pandoc', '-f', 'markdown', '-t', 'latex'],
                     input=text, capture_output=True, text=True, check=True).stdout
pathlib.Path('abstract.tex').write_text(tex)
EOF
echo "abstract.tex regenerated"

# refs_clean.bib = references.bib minus note fields (biblatex prints notes; ours are TODO markers)
python3 - <<'EOF'
import re, pathlib
bib = pathlib.Path('../chapters/references.bib').read_text()
bib = re.sub(r',?\s*note\s*=\s*\{[^{}]*\}', '', bib)
pathlib.Path('refs_clean.bib').write_text(bib)
EOF
echo "refs_clean.bib regenerated (note fields stripped)"

# word count (main-matter fragments only, per regulations it's the body count)
if command -v texcount >/dev/null 2>&1; then
  wc_total=$(texcount -total -sum -q $GENERATED 2>/dev/null | awk '/Sum count/{print $NF}')
else
  # approximation: strip TeX commands/math and count words (verify with texcount before submission)
  wc_total="$(sed -E -e 's/\\[a-zA-Z]+(\[[^]]*\])?(\{[^{}]*\})?//g' -e 's/[{}$&%]//g' $GENERATED \
    | wc -w | tr -d ' ') (approx.)"
fi
echo "\\newcommand{\\uobwordcount}{${wc_total:-TODO}}" > wordcount.tex
echo "word count: ${wc_total:-TODO}"

pdflatex -interaction=nonstopmode thesis.tex > build_school.log 2>&1 || {
  echo "pdflatex pass 1 FAILED — tail:"; grep -A3 '^!' build_school.log | head -30; exit 1; }
biber thesis >> build_school.log 2>&1 || {
  echo "biber FAILED — tail:"; tail -20 build_school.log; exit 1; }
pdflatex -interaction=nonstopmode thesis.tex >> build_school.log 2>&1 || true
pdflatex -interaction=nonstopmode thesis.tex >> build_school.log 2>&1 || {
  echo "pdflatex final pass FAILED — tail:"; grep -A3 '^!' build_school.log | head -30; exit 1; }

pages=$(pdfinfo thesis.pdf 2>/dev/null | awk '/Pages/{print $2}')
echo "OK: built thesis.pdf (${pages:-?} pages)"
# check the FINAL pass only (thesis.log), not the accumulated build_school.log
undef=$(grep -c 'Citation .* undefined' thesis.log || true)
[ "${undef:-0}" -gt 0 ] && echo "WARNING: $undef undefined citations" || echo "citations: all resolved"
