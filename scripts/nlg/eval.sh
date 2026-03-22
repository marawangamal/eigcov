
# Run eval on single model
olmes \
  --model pmahdavi/Llama-3.1-8B-math-reasoning \
  --task tulu_3_dev \
  --limit 1 \
  --output-dir ~/scratch/results


olmes \
  --model r-three/Llama-3.1-8B-merged-eigcov \
  --task tulu_3_dev_fast \
  --output-dir results-nlg


METHODS=(eigcov tsv isoc)
for FT_MODE in "${FT_MODES[@]}"; do
    olmes \
    --model pmahdavi/Llama-3.1-8B-merged-${METHOD} \
    --task tulu_3_dev \
    --limit 1 \
    --output-dir ~/scratch/results
done

export HF_HOME="$SCRATCH/huggingface"
olmes   --model r-three/Llama-3.1-8B-merged-eigcov  --task gsm8k::tulu drop::llama3 minerva_math::tulu codex_humaneval::tulu codex_humanevalplus::tulu ifeval::tulu popqa::tulu "bbh:cot-v1::tulu"   --output-dir results-nlg  --gpus 4


