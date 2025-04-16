#!/bin/bash

# Set paths and configurations
F5TTS_DIR="/PATH/TO/F5-TTS"
DISCRIMINATOR_DIR="/PATH/TO/DISCRIMINATOR"
F5TTS_CONFIG="F5TTS_Base_train.yaml"

# Number of GAN training loops
TOTAL_LOOPS=10


for ((i=3; i<=TOTAL_LOOPS; i++))
do
    echo "Starting GAN training loop $i"

    # Step 1: Train the F5TTS generator
    echo "Training F5TTS generator...loop $i"
    cd $F5TTS_DIR
    accelerate launch src/f5_tts/train/train.py --config-name $F5TTS_CONFIG
    if [ $? -ne 0 ]; then
        echo "F5TTS training failed. Exiting loop $i."
        exit 1
    fi

    # Step 2: Use the trained generator for inference
    echo "Running F5TTS inference...loop $i"
    python $F5TTS_DIR/src/f5_tts/infer/batch_infer_f5.py 
    if [ $? -ne 0 ]; then
        echo "F5TTS inference failed. Exiting loop $i."
        exit 1
    fi
    
    # Step 3: Train the classifier using the generated data
    echo "Training classifier...loop $i"
    cd $CLASSIFY_DIR
    python train.py
    if [ $? -ne 0 ]; then
        echo "Classifier training failed. Exiting loop $i."
        exit 1
    fi

    echo "Completed GAN training loop $i"
done

echo "GAN training finished for $TOTAL_LOOPS loops."
