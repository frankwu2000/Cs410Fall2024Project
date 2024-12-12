# Cs410Fall2024Project
CS410 text information system final project

Contributor: @hongyiw6, @yingche2

project proposal: https://textdata.org/submissions/67230f893d35dbf487544fee

project demo: https://mediaspace.illinois.edu/media/t/1_mgo5e42o

## Overview
This repository contains scripts and utilities to process CS410 lecture subtitles (in WebVTT format), segment them based on semantic similarity using Sentence Transformers, and evaluate the segmentation performance against labeled data. It includes preprocessing, encoding, segmentation, and evaluation steps.

## Files
1. main.py

    This is the primary script for processing subtitles, segmenting them, and evaluating the results. Key functionalities:

    - Preprocessing subtitles using SubtitleProcessor.
    - Encoding sentences with Sentence Transformers.
    - Identifying segmentation points using cosine similarity.
    - Evaluating segmentation accuracy against labeled data.

2. SubtitleProcessor.py
    
    Utility class for processing VTT files. 

    - Cleans subtitle text.
    - Combines consecutive captions into sentences based on punctuation.
    - Outputs a list of sentences with associated start times and indices.

3. W1-W6_subtitle_data
    Contains the CS410 lecture subtitles Week1 to Week6 Vtt file downloaded from Coursera

4. W1-W6_subtitle_data_labelled_segment_output
    
    Contains labeled segmentation points for evaluating the algorithm. This file is used as ground truth to calculate performance metrics.

    Currently only week1 has labelled data.

6. Makefile
    
    Automates tasks such as running the main script, cleaning temporary files, and setting up the environment.

## Folder Structure

- Input:
    - data_folder: Contains .vtt files for subtitle data.
- Output:
    - sentences_folder: Stores processed sentences as .txt files.
    - sbert_segments_output: Contains segmentation results in JSON format.

## Workflow
1. Preprocessing

    The SubtitleProcessor extracts and cleans subtitle text, creating sentences with start times and indices.

2. Encoding and Segmentation

    Sentences are encoded using SentenceTransformer.
A sliding window approach computes cosine similarity between adjacent groups of sentences.
Points with low similarity (below a threshold) are marked as segmentation points.

3. Evaluation

    Segmentation points are compared with labeled ground truth.
Metrics such as Accuracy, Precision, Recall, and F1 Score are calculated.

## Evaluation Metrics
- True Positive (TP): Correctly identified segmentation points.
- False Positive (FP): Incorrectly identified points.
- True Negative (TN): Correctly identified non-segmentation points.
- False Negative (FN): Missed segmentation points.

Derived Metrics:
- Accuracy: (TP + TN) / Total
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1 Score: 2 * (Precision * Recall) / (Precision + Recall)

## Notes
- Threshold: Adjust similarity_threshold_percentile for sensitivity.
- Customization: Modify the SubtitleProcessor for specific cleaning or sentence splitting logic.

This system is designed for lecture subtitle processing but can be adapted for other text segmentation tasks. For questions, please reach out to the repository maintainer.


## How to set up project
1. Install virtual environment
    ```
    pip install virtualenv
    ```

2. Create virtual environment
    ```
    python -m venv finalProject
    ```

3. Activate virtual environment
    ```
    source finalProject/bin/activate
    ```

4. Install dependency
    ```
    pip install -r requirements.txt
    ```

5. (To exit virtual environment) Deactivate virtual environment
    ```
    deactivate
    ```
## How to Build and Run the Project

1. **Run the Program**:
   - Execute the command:
     ```bash
     make build
     ```
   - This will generate an output file named `W1-W6_subtitle_data_sbert_segment_output`, containing the segmentation results. 
   - If the file already exists, running `make build` will reuse the existing file and proceed to evaluate the results against the labeled data.

2. **Regenerate the Output File**:
   - To create a new segmentation output, first clean up the previous results by running:
     ```bash
     make clean
     ```
   - Then, rerun the program with:
     ```bash
     make build
     ```