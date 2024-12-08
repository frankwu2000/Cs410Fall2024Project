import math
import time
from SubtitleProcessor import SubtitleProcessor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import tqdm
import json

def normalize_list(values):
    min_value = min(values)
    max_value = max(values)
    
    normalized_values = [(x - min_value) / (max_value - min_value) for x in values]
    
    return normalized_values

def encoding_sentences(model, window_size, threshold_factor, sentences: list, sentences_index: list):
    segmentation_points = []
    similarity_list = []
    comparison_points = []
    
    # Sliding window over sentences
    for i in range(len(sentences) - window_size):
        # Get embeddings for the current window and the next window
        window1 = sentences[i:i + window_size]
        window2 = sentences[i + 1:i + 1 + window_size]
        
        # Get embeddings for both windows
        embedding1 = np.mean(model.encode(window1), axis=0)
        embedding2 = np.mean(model.encode(window2), axis=0)
        
        # Compute cosine similarity between the two window embeddings
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        similarity_list.append(similarity)
        comparison_points.append(i)

    # normalize similarity
    similarity_list_norm = normalize_list(similarity_list)        
    threshold = sum(similarity_list_norm)/len(similarity_list_norm) * threshold_factor
    
    for i in range(len(comparison_points)):    
        # If similarity is below the threshold, mark the end of the first window as a segmentation point
        if similarity_list_norm[i] < threshold:
            segmentation_points.append((comparison_points[i], sentences_index[comparison_points[i]]))  

    return segmentation_points

def preprocess(vtt_file):
    # Initialize detector
    processor = SubtitleProcessor()
    sentences = processor.process_vtt(vtt_file)

    sentences_text = []
    sentences_index = []
    for i, sentence in enumerate(sentences, 1):  # enumerate from 1 for readability
        # print(f"\nSentence {i}:")
        # print(f"Start Time: {sentence['start_time']}")
        # print(f"Caption Index: {sentence['index']}")
        # print(f"{sentence['text']}")
        # print("-" * 80)  # Print a separator line between sentences
        sentences_text.append(sentence['text'])
        sentences_index.append(sentence['index'])
    return sentences_text, sentences_index

def load_sentences_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read all lines, stripping newline characters
        sentences = [line.strip() for line in file if line.strip()]
    return sentences

def load_segment_indexes_list(file_path):
    load_segment_indexes_list = []
    with open(file_path, "r") as file:
        for segment_indexes in file:
            load_segment_indexes_list.append(json.loads(segment_indexes))
    
    return load_segment_indexes_list

def save_segment_indexes_list(file_path, segment_indexes_list):
    with open(file_path, "w") as file:
        for segment_indexes in segment_indexes_list:
            file.write(str(segment_indexes) + "\n")

def main():
    # Path to the VTT file
    data_folder = "W1-W6_subtitle_data"
    vtt_file_list = os.listdir(data_folder)
    # TODO remove test file
    vtt_file_list = ["1.1.vtt","1.2.vtt","1.3.vtt"]
    sbert_model = SentenceTransformer('all-MPNet-base-v2')
    segment_indexes_list_path = data_folder+"_segment_indexes_list"

    print(f"# use SBert encoding and segment lecture files in {data_folder}")
    print(f"{'#' * 100}")

    # TODO: debug 'not'
    if not os.path.exists(segment_indexes_list_path):
        segment_indexes_list = load_segment_indexes_list(segment_indexes_list_path)
        print(segment_indexes_list)
    else:
        # TODO for vtt_file_name in tqdm.tqdm(vtt_file_list):
        segment_indexes_list = []
        for vtt_file_name in vtt_file_list:
            vtt_file_path = data_folder + "/" + vtt_file_name
            # vtt_file = "W1-W6_subtitle_data/1.2.vtt"

            # preprocess
            # print(f"# preprocess {vtt_file_path}")
            sentences, sentences_index = preprocess(vtt_file_path)

            # encoding and segmentation
            # print(f"# encoding and segmenting {vtt_file_path}")
            # TODO: test dynamic window size depends on total lines
            sliding_window_size = 2
            # TODO: test dynamic threshold size depends on total lines
            similarity_threshold_factor = 0.3
            segment_points = encoding_sentences(sbert_model, sliding_window_size, similarity_threshold_factor, sentences, sentences_index)

            # TODO: debug
            for point, point_index in segment_points:
                print(f"Segmentation point at sentence {point + 1}: {sentences[point]}, point_index: {point_index}")
            
            # Get index points from segmentation result
            segment_indexes = [sg[1] for sg in segment_points]
            print(f"segment_indexes: {segment_indexes}")
            segment_indexes_list.append(segment_indexes)

        save_segment_indexes_list(segment_indexes_list_path, segment_indexes_list)

    # Use LLM for benchmark

    # evaluation
    # compare segment against user annotated result
    # get precision, recall, F1-score

if __name__ == "__main__":
    main()