#!/bin/bash

# Set the test file prefix and output paths
TEST_FILE_PREFIX=phosphatase
RECORDS_FILE=${TEST_FILE_PREFIX}.json
CHECKPOINT_DIR=../experiments/km/mve/trainvaltest/seqemb36_attn6_esm_ens10/

# Create PDB records from the input CSV file
python ./scripts/create_pdbrecords.py --data_file ${TEST_FILE_PREFIX}.csv --out_file ${RECORDS_FILE}

# Make predictions using the pre-trained model
python predict.py --test_path ${TEST_FILE_PREFIX}.csv --preds_path ${TEST_FILE_PREFIX}_preds.csv --checkpoint_dir $CHECKPOINT_DIR --uncertainty_method mve --smiles_column SMILES --individual_ensemble_predictions --protein_records_path $RECORDS_FILE
