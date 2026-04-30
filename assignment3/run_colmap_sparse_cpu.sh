#!/bin/bash
set -e

DATASET_PATH=$1

if [ -z "$DATASET_PATH" ]; then
    echo "Usage: bash run_colmap_sparse_cpu.sh /path/to/data"
    exit 1
fi

echo "Dataset path: $DATASET_PATH"

# 清理旧结果，避免重复运行时混乱
rm -f "$DATASET_PATH/database.db"
rm -rf "$DATASET_PATH/sparse"

mkdir -p "$DATASET_PATH/sparse"

echo "Step 1: Feature extraction on CPU..."

colmap feature_extractor \
    --database_path "$DATASET_PATH/database.db" \
    --image_path "$DATASET_PATH/images" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model PINHOLE \
    --FeatureExtraction.use_gpu 0

echo "Step 2: Exhaustive matching on CPU..."

colmap exhaustive_matcher \
    --database_path "$DATASET_PATH/database.db" \
    --FeatureMatching.use_gpu 0

echo "Step 3: Sparse reconstruction / mapper..."

colmap mapper \
    --database_path "$DATASET_PATH/database.db" \
    --image_path "$DATASET_PATH/images" \
    --output_path "$DATASET_PATH/sparse"

echo "Step 4: Convert sparse model to PLY..."

colmap model_converter \
    --input_path "$DATASET_PATH/sparse/0" \
    --output_path "$DATASET_PATH/sparse/sparse.ply" \
    --output_type PLY

echo "Step 5: Convert sparse model to TXT..."

mkdir -p "$DATASET_PATH/sparse/sparse_txt"

colmap model_converter \
    --input_path "$DATASET_PATH/sparse/0" \
    --output_path "$DATASET_PATH/sparse/sparse_txt" \
    --output_type TXT

echo "Done."
echo "PLY output: $DATASET_PATH/sparse/sparse.ply"
echo "TXT output: $DATASET_PATH/sparse/sparse_txt"