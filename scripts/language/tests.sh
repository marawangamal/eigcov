#!/bin/bash
set -e

echo "=== Language module import tests ==="

python -c "from src.language.modeling import T5Wrapper; print('OK: T5Wrapper')"
python -c "from src.language.linearize import LinearizedT5Wrapper; print('OK: LinearizedT5Wrapper')"
python -c "from src.language.task_vectors import LanguageNonLinearTaskVector, LanguageLinearizedTaskVector; print('OK: Language task vectors')"
python -c "from src.language.datasets.dataset_readers import get_datasetReader; print('OK: dataset_readers')"
python -c "from src.language.datasets.dataset_mixture import get_datasetMixtureReader; print('OK: dataset_mixture')"
python -c "from src.language.datasets.batcher import Batcher; print('OK: Batcher')"
python -c "from src.language.datasets.pytorch_dataset import PytorchDataset; print('OK: PytorchDataset')"
python -c "from src.language.eval import eval_single_dataset, evaluate_task_vector; print('OK: eval')"
python -c "from src.language.eval.evaluator import Evaluator; print('OK: Evaluator')"
python -c "from src.language.eval.scorer import Scorer; print('OK: Scorer')"
python -c "from scripts.language.finetune import finetune; print('OK: finetune')"
python -c "from src.language.args import parse_arguments; print('OK: args')"
python -c "from src.language.utils import find_optimal_coef, DotDict; print('OK: utils')"

echo ""
echo "=== All imports passed ==="
