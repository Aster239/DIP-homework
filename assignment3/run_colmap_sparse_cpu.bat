@echo off
set DATASET_PATH=.\data

if exist "%DATASET_PATH%\database.db" del "%DATASET_PATH%\database.db"
if exist "%DATASET_PATH%\sparse" rmdir /s /q "%DATASET_PATH%\sparse"

mkdir "%DATASET_PATH%\sparse"

echo Step 1: Feature extraction on CPU...

colmap feature_extractor ^
    --database_path "%DATASET_PATH%\database.db" ^
    --image_path "%DATASET_PATH%\images" ^
    --ImageReader.single_camera 1 ^
    --ImageReader.camera_model PINHOLE ^
    --FeatureExtraction.use_gpu 0

echo Step 2: Exhaustive matching on CPU...

colmap exhaustive_matcher ^
    --database_path "%DATASET_PATH%\database.db" ^
    --FeatureMatching.use_gpu 0 ^
    --SiftMatching.guided_matching 1

echo Step 3: Sparse reconstruction...

colmap mapper ^
    --database_path "%DATASET_PATH%\database.db" ^
    --image_path "%DATASET_PATH%\images" ^
    --output_path "%DATASET_PATH%\sparse"

echo Step 4: Convert sparse model to PLY...

colmap model_converter ^
    --input_path "%DATASET_PATH%\sparse\0" ^
    --output_path "%DATASET_PATH%\sparse\sparse.ply" ^
    --output_type PLY

echo Step 5: Convert sparse model to TXT...

mkdir "%DATASET_PATH%\sparse\sparse_txt"

colmap model_converter ^
    --input_path "%DATASET_PATH%\sparse\0" ^
    --output_path "%DATASET_PATH%\sparse\sparse_txt" ^
    --output_type TXT

echo Done.
pause