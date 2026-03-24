#!/usr/bin/env python3
"""
Script para baixar e converter o dataset Bosque (Universal Dependencies)
para os formatos esperados pelo LAL-Parser:
  - Constituency: formato bracketed PTB (uma árvore por linha)
  - Dependency: formato CoNLL-X (tab-separated, 10 colunas)

O Bosque UD não possui anotação de constituência nativa.
Portanto, criamos árvores de constituência "flat" a partir das POS tags,
que são suficientes para o treinamento do LAL-Parser com foco em dependência.

Uso:
  python scripts/prepare_bosque.py [--output-dir data/bosque]
"""

import os
import sys
import argparse
import urllib.request
import tarfile
import shutil


BOSQUE_UD_URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_Portuguese-Bosque/master"
BOSQUE_FILES = {
    "train": "pt_bosque-ud-train.conllu",
    "dev": "pt_bosque-ud-dev.conllu",
    "test": "pt_bosque-ud-test.conllu",
}

# Mapeamento de UPOS (Universal POS) para POS tags estilo PTB
# O LAL-Parser espera tags no estilo Penn Treebank
UPOS_TO_PTB = {
    "ADJ": "JJ",
    "ADP": "IN",
    "ADV": "RB",
    "AUX": "MD",
    "CCONJ": "CC",
    "DET": "DT",
    "INTJ": "UH",
    "NOUN": "NN",
    "NUM": "CD",
    "PART": "RP",
    "PRON": "PRP",
    "PROPN": "NNP",
    "PUNCT": ".",
    "SCONJ": "IN",
    "SYM": "SYM",
    "VERB": "VB",
    "X": "FW",
    "_": "XX",
}


def download_file(url, dest_path):
    """Baixa um arquivo da URL para dest_path."""
    print(f"  Baixando {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        print(f"  ERRO ao baixar: {e}")
        return False


def parse_conllu(filepath):
    """
    Lê um arquivo CoNLL-U e retorna lista de sentenças.
    Cada sentença é uma lista de tokens (dicionários).
    Ignora linhas de comentário e tokens com IDs compostos (ex: 1-2).
    """
    sentences = []
    current_sentence = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) != 10:
                continue
            # Ignorar tokens compostos (ex: "1-2") e tokens nulos (ex: "1.1")
            token_id = parts[0]
            if '-' in token_id or '.' in token_id:
                continue

            token = {
                'id': int(parts[0]),
                'form': parts[1],
                'lemma': parts[2],
                'upos': parts[3],
                'xpos': parts[4] if parts[4] != '_' else parts[3],
                'feats': parts[5],
                'head': int(parts[6]) if parts[6] != '_' else 0,
                'deprel': parts[7],
                'deps': parts[8],
                'misc': parts[9],
            }
            current_sentence.append(token)

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def convert_to_conllx(sentences, output_path):
    """
    Converte sentenças para formato CoNLL-X (10 colunas, tab-separated).
    Formato: ID FORM LEMMA CPOSTAG POSTAG FEATS HEAD DEPREL PHEAD PDEPREL
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for sent in sentences:
            for token in sent:
                ptb_pos = UPOS_TO_PTB.get(token['upos'], 'XX')
                # CoNLL-X: ID FORM LEMMA CPOSTAG POSTAG FEATS HEAD DEPREL PHEAD PDEPREL
                line = '\t'.join([
                    str(token['id']),
                    token['form'],
                    '_',               # LEMMA (placeholder)
                    ptb_pos,            # CPOSTAG (POS tag estilo PTB)
                    ptb_pos,            # POSTAG (mesmo)
                    '_',                # FEATS
                    str(token['head']),  # HEAD
                    token['deprel'],     # DEPREL
                    '_',                # PHEAD
                    '_',                # PDEPREL
                ])
                f.write(line + '\n')
            f.write('\n')  # Linha vazia entre sentenças


def convert_to_constituency(sentences, output_path):
    """
    Cria árvores de constituência "flat" a partir das POS tags.
    Formato: (TOP (S (POS word) (POS word) ...))

    NOTA: Como o Bosque UD não possui anotação de constituência nativa,
    geramos árvores flat. Isso é suficiente para o LAL-Parser porque o
    foco do TCC é análise de DEPENDÊNCIA — as árvores de constituência
    são usadas pelo framework mas o score principal é UAS/LAS.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for sent in sentences:
            # Filtra tokens de pontuação para a árvore de constituência
            leaves = []
            for token in sent:
                ptb_pos = UPOS_TO_PTB.get(token['upos'], 'XX')
                word = token['form']
                # Escape de parênteses no formato PTB
                word = word.replace('(', '-LRB-').replace(')', '-RRB-')
                leaves.append(f"({ptb_pos} {word})")

            if leaves:
                # Encapsula as folhas em um nó X dentro de S
                tree = "(TOP (S (X " + " ".join(leaves) + ")))"
                f.write(tree + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Prepara o dataset Bosque UD para o LAL-Parser"
    )
    parser.add_argument(
        '--output-dir', default='data/bosque',
        help='Diretório de saída para os dados convertidos'
    )
    parser.add_argument(
        '--conllu-dir', default=None,
        help='Diretório com arquivos .conllu já baixados (opcional)'
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    download_dir = os.path.join(output_dir, 'raw')
    os.makedirs(download_dir, exist_ok=True)

    # Passo 1: Baixar ou localizar os arquivos CoNLL-U
    print("=" * 60)
    print("PASSO 1: Obtendo dados do Bosque UD")
    print("=" * 60)

    conllu_files = {}
    for split, filename in BOSQUE_FILES.items():
        if args.conllu_dir:
            src = os.path.join(args.conllu_dir, filename)
            if os.path.exists(src):
                conllu_files[split] = src
                print(f"  [{split}] Encontrado: {src}")
            else:
                print(f"  [{split}] ERRO: Não encontrado em {src}")
                sys.exit(1)
        else:
            dest = os.path.join(download_dir, filename)
            if os.path.exists(dest):
                print(f"  [{split}] Já existe: {dest}")
                conllu_files[split] = dest
            else:
                url = f"{BOSQUE_UD_URL}/{filename}"
                if download_file(url, dest):
                    conllu_files[split] = dest
                else:
                    print(f"  ERRO: Não foi possível baixar {filename}")
                    print(f"  Tente baixar manualmente de:")
                    print(f"    https://github.com/UniversalDependencies/UD_Portuguese-Bosque")
                    sys.exit(1)

    # Passo 2: Converter para formatos do LAL-Parser
    print()
    print("=" * 60)
    print("PASSO 2: Convertendo para formatos do LAL-Parser")
    print("=" * 60)

    for split in ['train', 'dev', 'test']:
        if split not in conllu_files:
            continue

        print(f"\n  Processando {split}...")
        sentences = parse_conllu(conllu_files[split])
        print(f"    {len(sentences)} sentenças lidas")

        # Dependency (CoNLL-X)
        dep_output = os.path.join(output_dir, f'bosque_{split}.conllx')
        convert_to_conllx(sentences, dep_output)
        print(f"    Dependência: {dep_output}")

        # Constituency (PTB bracketed)
        const_output = os.path.join(output_dir, f'bosque_{split}.clean')
        convert_to_constituency(sentences, const_output)
        print(f"    Constituência: {const_output}")

    # Passo 3: Verificação
    print()
    print("=" * 60)
    print("PASSO 3: Verificação")
    print("=" * 60)

    for split in ['train', 'dev', 'test']:
        dep_file = os.path.join(output_dir, f'bosque_{split}.conllx')
        const_file = os.path.join(output_dir, f'bosque_{split}.clean')
        if os.path.exists(dep_file) and os.path.exists(const_file):
            # Contar sentenças
            with open(dep_file) as f:
                dep_sents = sum(1 for line in f if line.strip() == '')
            with open(const_file) as f:
                const_sents = sum(1 for line in f if line.strip())
            print(f"  [{split}] dep={dep_sents} sentenças, const={const_sents} sentenças", end='')
            if dep_sents == const_sents:
                print(" ✓")
            else:
                print(" ✗ INCONSISTÊNCIA!")

    print()
    print("=" * 60)
    print("CONCLUÍDO!")
    print(f"Dados salvos em: {output_dir}/")
    print()
    print("Arquivos gerados:")
    print(f"  Treino (const):  {output_dir}/bosque_train.clean")
    print(f"  Treino (dep):    {output_dir}/bosque_train.conllx")
    print(f"  Dev (const):     {output_dir}/bosque_dev.clean")
    print(f"  Dev (dep):       {output_dir}/bosque_dev.conllx")
    print(f"  Teste (const):   {output_dir}/bosque_test.clean")
    print(f"  Teste (dep):     {output_dir}/bosque_test.conllx")
    print("=" * 60)


if __name__ == '__main__':
    main()
