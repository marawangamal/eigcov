
# Run eval on single model
olmes \
  --model pmahdavi/Llama-3.1-8B-math-reasoning \
  --task tulu_3_dev \
  --limit 1 \
  --output-dir ~/scratch/results


METHODS=(eigcov tsv isoc)
for FT_MODE in "${FT_MODES[@]}"; do
    olmes \
    --model pmahdavi/Llama-3.1-8B-merged-${METHOD} \
    --task tulu_3_dev \
    --limit 1 \
    --output-dir ~/scratch/results
done


# olmes   --model pmahdavi/Llama-3.1-8B-math-reasoning   --task gsm8k::tulu drop::llama3 minerva_math::tulu codex_humaneval::tulu codex_humanevalplus::tulu ifeval::tulu popqa::tulu "bbh:cot-v1::tulu"   --task-args '{"limit": 2}'   --output-dir ~/scratch/results-nlg 