#!/usr/bin/env bash
# ============================================================================
# Script de treinamento do LAL-Parser com BERTimbau + Bosque UD
#
# TCC: "Desenvolvimento de um Modelo Transformer Encoder para Análise de
#       Dependência na Língua Portuguesa"
# Autor: Leonardo Gonçalves Marotta
#
# Uso: bash train_bosque_bertimbau.sh
# ============================================================================

# --- Configuração ---
# BERTimbau Base (768 dim, 12 layers, 12 heads, ~110M params)
# Cabe em GPU com ~6GB VRAM. No Colab Free (T4 16GB) roda confortavelmente.
BERT_MODEL="neuralmind/bert-base-portuguese-cased"

# Diretórios de dados (gerados por scripts/prepare_bosque.py)
TRAIN_CONST="data/bosque/bosque_train.clean"
DEV_CONST="data/bosque/bosque_dev.clean"
TRAIN_DEP="data/bosque/bosque_train.conllx"
DEV_DEP="data/bosque/bosque_dev.conllx"

# Diretório do modelo
MODEL_DIR="models"
MODEL_NAME="lal_bertimbau_bosque"

# --- Verificar dados ---
if [ ! -f "$TRAIN_CONST" ]; then
    echo "ERRO: Dados do Bosque não encontrados em data/bosque/"
    echo "Execute primeiro: python scripts/prepare_bosque.py"
    exit 1
fi

mkdir -p "$MODEL_DIR"

# --- Treinar ---
python src_joint/main.py train \
    --model-path-base "${MODEL_DIR}/${MODEL_NAME}" \
    --epochs 50 \
    --use-bert \
    --bert-model "$BERT_MODEL" \
    --no-bert-do-lower-case \
    --use-tags \
    --const-lada 0.5 \
    --dataset ptb \
    --embedding-path data/glove.gz \
    --model-name "$MODEL_NAME" \
    --checks-per-epoch 4 \
    --num-layers 2 \
    --learning-rate 0.00005 \
    --batch-size 32 \
    --eval-batch-size 16 \
    --subbatch-max-tokens 500 \
    --train-ptb-path "$TRAIN_CONST" \
    --dev-ptb-path "$DEV_CONST" \
    --dep-train-ptb-path "$TRAIN_DEP" \
    --dep-dev-ptb-path "$DEV_DEP" \
    --lal-d-kv 128 \
    --lal-d-proj 128 \
    --no-lal-resdrop
