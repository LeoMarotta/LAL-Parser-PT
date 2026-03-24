#!/usr/bin/env bash
# ============================================================================
# Script de teste do modelo treinado com Bosque UD
# ============================================================================

MODEL_PATH="models/lal_bertimbau_bosque_dev=*"

# Pegar o modelo com melhor score (último salvo)
BEST_MODEL=$(ls -t ${MODEL_PATH} 2>/dev/null | head -1)

if [ -z "$BEST_MODEL" ]; then
    echo "ERRO: Nenhum modelo treinado encontrado em models/"
    echo "Treine primeiro com: bash train_bosque_bertimbau.sh"
    exit 1
fi

echo "Usando modelo: $BEST_MODEL"

python src_joint/main.py test \
    --dataset ptb \
    --eval-batch-size 8 \
    --consttest-ptb-path data/bosque/bosque_test.clean \
    --deptest-ptb-path data/bosque/bosque_test.conllx \
    --embedding-path data/glove.gz \
    --model-path-base "$BEST_MODEL"
