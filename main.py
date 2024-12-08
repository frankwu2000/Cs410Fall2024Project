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

def encoding_sentences(model, sentences: list, sentences_index: list):
    sliding_window_size = 5
    similarity_threshold_percentile = 5
    
    segmentation_points = []
    similarity_list = []
    comparison_points = []
    
    # Sliding window over sentences
    for i in range(len(sentences) - sliding_window_size):
        # Get embeddings for the current window and the next window
        window1 = sentences[i:i + sliding_window_size]
        window1 = ["".join(window1)]
        window2 = sentences[i + 1:i + 1 + sliding_window_size]
        window2 = ["".join(window2)]

        # Get embeddings for both windows
        embedding1 = np.mean(model.encode(window1), axis=0)
        embedding2 = np.mean(model.encode(window2), axis=0)
        
        # Compute cosine similarity between the two window embeddings
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        similarity_list.append(similarity)
        comparison_points.append(i)

    # Normalize similarity by min and max value. 
    # Use percentile to get threshold
    similarity_list_norm = normalize_list(similarity_list)        
    threshold = np.percentile(similarity_list_norm, similarity_threshold_percentile)
    
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

def evaluate(test_list: list, golden_standard_list: list):
    threshold = 3
    truePositive = 0
    
    for r1 in test_list:
        for r2 in golden_standard_list:
            if np.abs(r1-r2) <= threshold:
                truePositive += 1
                break
    
    accuracy = truePositive / len(golden_standard_list)

    return accuracy    



def main():
    # Path to the VTT file
    data_folder = "W1-W6_subtitle_data"
    sbert_model = SentenceTransformer('all-MPNet-base-v2')
    
    vtt_file_list = sorted(os.listdir(data_folder))

    # TODO remove test file
    # vtt_file_list = ["1.1.vtt","1.2.vtt","1.3.vtt"]
    
    sbert_segments_output_path = data_folder + "_sbert_segment_output"

    print(f"# use SBert encoding and segment lecture files in {data_folder}")
    print(f"{'#' * 100}")

    sbert_segments = []
    if os.path.exists(sbert_segments_output_path):
        sbert_segments = load_segment_indexes_list(sbert_segments_output_path)
        # print(sbert_segments_output_path)
    else:
        for vtt_file_name in tqdm.tqdm(vtt_file_list):
        # for vtt_file_name in vtt_file_list:
        
            vtt_file_path = data_folder + "/" + vtt_file_name

            # preprocess
            sentences, sentences_index = preprocess(vtt_file_path)

            # encoding and segmentation
            segment_points = encoding_sentences(sbert_model, sentences, sentences_index)

            # for point, point_index in segment_points:
                # print(f"Segmentation point at sentence {point + 1}: {sentences[point]}, point_index: {point_index}")
            
            # Get index points from segmentation result
            segment_indexes = [sg[1] for sg in segment_points]

            # print(f"vtt_file_name:{vtt_file_name}, segment_indexes: {segment_indexes}")
            sbert_segments.append(segment_indexes)

        save_segment_indexes_list(sbert_segments_output_path, sbert_segments)

    # LLM benchmark segments
    llm_labelled_segments = load_segment_indexes_list(data_folder+"_llm_labelled_segment_output")

    # evaluation
    print(f"#evaluate Sbert segmentation performance: # true positive / # true")
    accuracy_list = []
    for i in range(len(sbert_segments)):        
        accuracy = evaluate(sbert_segments[i],llm_labelled_segments[i])
        accuracy_list.append(accuracy)
    print(f"accuracy_list: {accuracy_list}")
    print(f"accuracy_list mean: {np.mean(accuracy_list)}")

    # compare segment against user annotated result
    # get precision, recall, F1-score

if __name__ == "__main__":
    main()