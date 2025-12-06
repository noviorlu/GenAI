#!/bin/bash

trained_model_name=pythia-160m-1024-marco-docs-bow-contrastive-pretrain-marco-passage-sft
EMBEDDING_OUTPUT_DIR=./data/embeddings/$trained_model_name/fiqa
QRELS_FILE=./data/datasets/fiqa/fiqa/qrels/test_trec.tsv

# Check if embeddings exist
if [ ! -f "$EMBEDDING_OUTPUT_DIR/query-test.pkl" ]; then
    echo "Error: Query embeddings not found at $EMBEDDING_OUTPUT_DIR/query-test.pkl"
    echo "Please run the encoding script first: bash encode_fiqa.sh"
    exit 1
fi

if [ ! -f "$EMBEDDING_OUTPUT_DIR/corpus.pkl" ]; then
    echo "Error: Corpus embeddings not found at $EMBEDDING_OUTPUT_DIR/corpus.pkl"
    echo "Please run the encoding script first: bash encode_fiqa.sh"
    exit 1
fi

# Convert qrels to TREC format if needed
if [ ! -f "$QRELS_FILE" ]; then
    echo "Converting qrels to TREC format..."
    python -c "
input_file = './data/datasets/fiqa/fiqa/qrels/test.tsv'
output_file = './data/datasets/fiqa/fiqa/qrels/test_trec.tsv'

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line_num, line in enumerate(f_in):
        parts = line.strip().split('\t')
        
        # Skip header row (usually first line with non-numeric data)
        if line_num == 0 and ('query' in line.lower() or 'qid' in line.lower() or not parts[-1].replace('.','').replace('-','').isdigit()):
            continue
            
        if len(parts) == 3:
            qid, docid, relevance = parts
            # Only process if relevance is numeric
            try:
                float(relevance)  # Test if it's a valid number
                f_out.write(f'{qid}\t0\t{docid}\t{relevance}\n')
            except ValueError:
                continue  # Skip non-numeric relevance scores
        elif len(parts) == 4:
            f_out.write(line)
print('Converted qrels to TREC format')
"
fi

echo "Using embeddings from: $EMBEDDING_OUTPUT_DIR"
echo "Using qrels from: $QRELS_FILE"
echo "Running in CPU-only mode to avoid GPU hanging..."

# Force CPU-only mode by setting CUDA_VISIBLE_DEVICES to empty
CUDA_VISIBLE_DEVICES="" python -m driver.search \
    --query_reps $EMBEDDING_OUTPUT_DIR/query-test.pkl \
    --passage_reps $EMBEDDING_OUTPUT_DIR/corpus.pkl \
    --depth 1000 \
    --batch_size 64 \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/run.test.txt \
    --save_metrics_to $EMBEDDING_OUTPUT_DIR/trec_eval_results.txt \
    --qrels $QRELS_FILE

# Check if results were generated
if [ -f "$EMBEDDING_OUTPUT_DIR/trec_eval_results.txt" ]; then
    echo "Search completed successfully!"
    echo "Results:"
    cat $EMBEDDING_OUTPUT_DIR/trec_eval_results.txt
else
    echo "Search failed - no results file generated"
    exit 1
fi