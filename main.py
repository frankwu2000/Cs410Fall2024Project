import math
import sys
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
            file.write(json.dumps(segment_indexes) + "\n")

def save_sentences(file_path, sentences):
    with open(file_path, "w") as file:
        for sentence in sentences:
            file.write(str(sentence) + "\n")

def load_sentences_file(folder_path):
    sentences_list = []
    sentences_file_list = sorted(os.listdir(folder_path))
    for sentence_file_path in sentences_file_list:
        with open(folder_path + "/" + sentence_file_path, "r") as file:
            sentences = []
            for sentence in file:
                sentences.append(sentence)
            sentences_list.append(sentences)
    return sentences_list

def evaluate(test_list: list, labeled_list: list, sentences_length: int, threshold = 0):
    truePositive = 0
    trueNegative = 0
    for r1 in test_list:
        for r2 in labeled_list:
            if np.abs(int(r1)-int(r2)) <= threshold:
                truePositive += 1
                break
    for i in range(1,sentences_length+1):
        if i not in test_list and i not in labeled_list:
            trueNegative += 1
    
    falsePositive = len(test_list) -  truePositive
    falseNegative = len(labeled_list) - truePositive

    return truePositive, trueNegative, falsePositive, falseNegative  

def main():
    # Path to the VTT file
    data_folder = sys.argv[1] + "_subtitle_data"
    sentences_folder = sys.argv[1] + "_sentences_data"

    sbert_model = SentenceTransformer('all-MPNet-base-v2')
    
    vtt_file_list = sorted(os.listdir(data_folder))
    
    sbert_segments_output_path = data_folder + "_sbert_segment_output"

    print(f"# use SBert encoding to segment lecture files in {data_folder}")
    print(f"{'#' * 100}")

    sbert_segments = []
    sentences_list = []
    if os.path.exists(sbert_segments_output_path):
        sentences_list = load_sentences_file(sentences_folder)
        sbert_segments = load_segment_indexes_list(sbert_segments_output_path)
    else:
        if not os.path.exists(sentences_folder):
            os.makedirs(sentences_folder)
        for vtt_file_name in tqdm.tqdm(vtt_file_list):
            vtt_file_path = data_folder + "/" + vtt_file_name

            # preprocess
            sentences, sentences_index = preprocess(vtt_file_path)
            sentences_list.append(sentences)
            save_sentences(sentences_folder + "/" + vtt_file_name + "_sentences.txt", sentences)

            # encoding and segmentation
            segment_points = encoding_sentences(sbert_model, sentences, sentences_index)

            for point, point_index in segment_points:
                print(f"{vtt_file_name}: Segmentation point at sentence {point + 1}: {sentences[point]}, point_index: {point_index}")
            
            # Get index points from segmentation result
            segment_indexes = [point+1 for point, point_index in segment_points]
            sbert_segments.append({vtt_file_name: segment_indexes})

    # Save segment indexes list
    save_segment_indexes_list(sbert_segments_output_path, sbert_segments)

    # labelled segments
    labelled_segments = load_segment_indexes_list(data_folder+"_labelled_segment_output")

    # evaluation
    print(f"#evaluate Sbert segmentation performance: # true positive / # true")
    sentences_len_list = [len(sentences) for sentences in sentences_list]
    eval_list = []
    decimal_place = 5
    for i in range(len(sbert_segments)):
        vtt_file_name = list(sbert_segments[i].keys())[0]
        sbert_segment = sbert_segments[i][vtt_file_name]
        labelled_segment = labelled_segments[i][vtt_file_name]
        if not labelled_segment:
            continue
        tp, tn, fp, fn = evaluate(sbert_segment,labelled_segment, sentences_len_list[i], threshold=4)
        accuracy, precision, recall, f1 = 0,0,0,0
        if (tp+tn+fp+fn) != 0:
            accuracy = round((tp + tn) / (tp+tn+fp+fn),decimal_place)
        if tp + fn != 0 and tp + fn != 0:
            precision = round(tp / (tp + fp),decimal_place)
            recall = round(tp / (tp + fn),decimal_place)
        if precision + recall != 0:
            f1 = round(2 * precision * recall / (precision + recall),decimal_place)
        eval = {"vtt_file_name": vtt_file_name, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        print(eval)
        eval_list.append(eval)

if __name__ == "__main__":
    main()