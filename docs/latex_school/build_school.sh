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
# -X utf8 is load-bearing: without it, Python decodes this heredoc, the pipe to
# pandoc, and the output file using the system locale (cp936 on a Chinese Windows),
# which silently turned the abstract's em dashes and ">=" into mojibake.
python3 -X utf8 - <<'EOF'
import subprocess, pathlib
text = '''Video is widely recorded but remains hard to search at the moment level. The questions people ask of footage are causal and temporal — *why* did the boy carry the present to the sofa — while the metadata and speech transcripts video search relies on are neither. This dissertation asks how far an agentic retrieval-augmented generation system can close that gap with off-the-shelf components on consumer hardware, treating every design choice as something to measure rather than assert.

The system indexes visual and transcript evidence as overlapping 8-second segments, rewrites a question into the scene-caption register a contrastive encoder can match, searches and re-ranks candidate moments, and answers through a bounded LangGraph agent that cites the seconds of video supporting each claim. Retrieval is scored on 3,358 temporally grounded NExT-QA/NExT-GQA questions over 567 videos under a deliberately strict criterion — correct video *and* tIoU ≥ 0.5 — which makes absolute recall small by construction and puts the weight on controlled comparisons.

- A reproducible, temporally grounded video RAG pipeline and evaluation harness, runnable end to end on one 6 GB laptop GPU (Chapters 3–4).
- A component-wise ablation over all 3,358 questions: a SigLIP index with query decomposition nearly doubles top-1 moment recall (0.026 to 0.048); decomposition buys recall at depth, visual re-ranking top-rank precision (§5.3).
- Two negative results with one diagnosis — late fusion never beats visual-only at any weight, and a text cross-encoder re-ranker degrades ranking, both traced to sparse, multilingual, conversational speech — together with a cross-domain replication on speech-dense video that reverses both exactly as the diagnosis predicts, making the modality ordering domain-conditional and measurable (§5.3.1, §5.3.3, §5.5).
- A two-stage equivalence finding: re-ranking a cheap index's top-30 with a stronger backbone matches indexing the whole corpus with it, decoupling quality from indexing cost (§5.3.4).
- A calibrated account of agentic answering: the graph agent lifts five-choice accuracy 0.447 to 0.547 and keyframe evidence lifts temporal-next accuracy 0.341 to 0.636, while prior-only baselines (0.560) leave the multimodal stack the only configuration that beats the bare model (§5.4).'''
tex = subprocess.run(['pandoc', '-f', 'markdown', '-t', 'latex'],
                     input=text, capture_output=True, text=True, check=True,
                     encoding='utf-8').stdout
pathlib.Path('abstract.tex').write_text(tex, encoding='utf-8')
EOF
echo "abstract.tex regenerated"

# refs_clean.bib = references.bib minus note fields (biblatex prints notes; ours are TODO markers)
python3 -X utf8 - <<'EOF'
import re, pathlib
bib = pathlib.Path('../chapters/references.bib').read_text(encoding='utf-8')
bib = re.sub(r',?\s*note\s*=\s*\{[^{}]*\}', '', bib)
pathlib.Path('refs_clean.bib').write_text(bib, encoding='utf-8')
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
