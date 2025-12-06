#!/bin/bash

test_data=fiqa
trained_model_name=pythia-160m-1024-marco-docs-bow-contrastive-pretrain-marco-passage-sft
EMBEDDING_OUTPUT_DIR=./data/embeddings/$trained_model_name/$test_data
trained_model_path=./data/model/$trained_model_name
data_dir=./data/datasets/fiqa/fiqa

# Create directories
mkdir -p $EMBEDDING_OUTPUT_DIR
mkdir -p ./data/datasets/fiqa

# Check if dataset exists, if not download it
if [ ! -d "$data_dir" ]; then
    echo "Dataset not found. Downloading FiQA dataset..."
    
    cd ./data/datasets/fiqa
    
    # Download FiQA dataset directly from BeIR
    echo "Downloading FiQA dataset..."
    wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip
    
    # Extract the dataset
    echo "Extracting dataset..."
    unzip -o fiqa.zip
    
    # Clean up zip file
    rm fiqa.zip
    
    echo "Dataset downloaded and extracted!"
    echo "Sample corpus entry:"
    head -n 1 fiqa/corpus.jsonl
    echo "Sample query entry:"
    head -n 1 fiqa/queries.jsonl
    
    # Return to original directory
    cd - > /dev/null
else
    echo "Dataset already exists at: $data_dir"
fi

# Encode queries using local JSONL files
echo "Encoding queries..."
python -m driver.encode \
  --output_dir=temp \
  --model_name_or_path $trained_model_path \
  --bf16 \
  --encode_is_query \
  --per_device_eval_batch_size 128 \
  --query_max_len 32 \
  --passage_max_len 128 \
  --dataset_path $data_dir/queries.jsonl \
  --dataset_split None \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/query-test.pkl

# Encode corpus using local JSONL files
echo "Encoding corpus..."
python -m driver.encode \
  --output_dir=temp \
  --model_name_or_path $trained_model_path \
  --bf16 \
  --per_device_eval_batch_size 128 \
  --query_max_len 32 \
  --passage_max_len 128 \
  --dataset_path $data_dir/corpus.jsonl \
  --dataset_number_of_shards 1 \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/corpus.pkl

echo "Encoding complete! Outputs saved to: $EMBEDDING_OUTPUT_DIR"