import math
import time
from SubtitleProcessor import SubtitleProcessor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import tqdm
import json

def encoding_sentences(model, sentences: list, sentences_index: list):
    sliding_window_size = 4
    similarity_threshold_percentile = 5
    
    segmentation_points = []
    similarity_list = []
    newSeg_points = []
    
    # Sliding window over sentences
    for i in range(len(sentences) - sliding_window_size):
        new_seg_index=i+1
        # Get embeddings for the current window and the next window
        window1 = sentences[i:i + sliding_window_size]
        window2 = sentences[new_seg_index:new_seg_index + sliding_window_size]

        # Get embeddings for both windows and apply average pooling
        embedding1 = np.mean(model.encode(window1), axis=0)
        embedding2 = np.mean(model.encode(window2), axis=0)
        
        # Compute cosine similarity between the two window embeddings
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        similarity_list.append(similarity)

        # add the first line of new segment as segment points
        newSeg_points.append(new_seg_index)

    # Use percentile to get threshold     
    threshold = np.percentile(similarity_list, similarity_threshold_percentile)
    
    for i in range(len(newSeg_points)):    
        # If similarity is below the threshold, mark the end of the first window as a segmentation point
        if similarity_list[i] < threshold:
            segmentation_points.append((newSeg_points[i], sentences_index[newSeg_points[i]]))  

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

def save_sentences(file_path, sentences):
    with open(file_path, "w") as file:
        for sentence in sentences:
            file.write(str(sentence) + "\n")

def evaluate(test_list: list, labeled_list: list, threshold = 0):
    truePositive = 0
    
    for r1 in test_list:
        for r2 in labeled_list:
            if np.abs(r1-r2) <= threshold:
                truePositive += 1
                break
    
    falsePositive = len(test_list) -  truePositive
    falseNegative = len(labeled_list) - truePositive

    return truePositive, falsePositive, falseNegative  

def main():
    # Path to the VTT file
    # data_folder = "W1-W6_subtitle_data"
    # TODO test
    data_folder = "W1_subtitle_data"
    sentences_folder = "W1_sentences_folder"
    sbert_model = SentenceTransformer('all-MPNet-base-v2')
    
    vtt_file_list = sorted(os.listdir(data_folder))
    
    sbert_segments_output_path = data_folder + "_sbert_segment_output"

    print(f"# use SBert encoding to segment lecture files in {data_folder}")
    print(f"{'#' * 100}")

    sbert_segments = []
    # TODO: test
    if os.path.exists(sbert_segments_output_path):
        sbert_segments = load_segment_indexes_list(sbert_segments_output_path)
    else:
        for vtt_file_name in tqdm.tqdm(vtt_file_list):
        # for vtt_file_name in vtt_file_list:
        
            vtt_file_path = data_folder + "/" + vtt_file_name

            # preprocess
            sentences, sentences_index = preprocess(vtt_file_path)

            save_sentences(sentences_folder + "/" + vtt_file_name + "_sentences.txt", sentences)

            # encoding and segmentation
            segment_points = encoding_sentences(sbert_model, sentences, sentences_index)

            for point, point_index in segment_points:
                print(f"Segmentation point at sentence {point + 1}: {sentences[point]}, point_index: {point_index}")
            
            # Get index points from segmentation result
            segment_indexes = [point+1 for point, point_index in segment_points]
            sbert_segments.append(segment_indexes)

        save_segment_indexes_list(sbert_segments_output_path, sbert_segments)

    # LLM benchmark segments
    # llm_labelled_segments = load_segment_indexes_list(data_folder+"_llm_labelled_segment_output")
    # TODO test
    llm_labelled_segments = load_segment_indexes_list(data_folder+"_llm_labelled_segment_output")

    # evaluation
    print(f"#evaluate Sbert segmentation performance: # true positive / # true")
    eval_list = []
    for i in range(len(sbert_segments)):        
        tp, fp, fn = evaluate(sbert_segments[i],llm_labelled_segments[i], threshold=4)
        precision, recall, f1 = 0,0,0
        if tp + fn != 0 and tp + fn != 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        eval = {"precision": precision, "recall": recall, "f1": f1}
        print(eval)
        eval_list.append(eval)

if __name__ == "__main__":
    main()