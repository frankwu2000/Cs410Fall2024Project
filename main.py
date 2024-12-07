import math
from SubtitleProcessor import SubtitleProcessor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import numpy as np

def normalize_list(values):
    # Find the min and max of the list
    min_value = min(values)
    max_value = max(values)
    
    # Apply min-max normalization
    normalized_values = [(x - min_value) / (max_value - min_value) for x in values]
    
    return normalized_values

def process_sentence(sentences: list):
    # Step 1: Sentence Encoding with SBERT
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Parameters
    # window_size = math.floor(len(sentences)/10)  # Sliding window size (N)
    window_size = 5
    # threshold = 0.5 # Cosine similarity threshold to mark segmentation
    print(f"window_size: {window_size}")

    # Find segmentation points
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
        # print(similarity)
        similarity_list.append(similarity)
        comparison_points.append(i)

    # normalize similarity
    similarity_list_norm = normalize_list(similarity_list)        
    threshold = sum(similarity_list_norm)/len(similarity_list_norm) * 0.5
    print(f"threshold: {threshold}")

    for i in range(len(comparison_points)):    
        # If similarity is below the threshold, mark the end of the first window as a segmentation point
        # print(f"similarity_list_norm[i]:{similarity_list_norm[i]}, threshold: {threshold}")
        if similarity_list_norm[i] < threshold:
            # print(similarity, threshold)
            segmentation_points.append(comparison_points[i])  # Mark the last sentence of the first window

    # Output the segmentation points (indices of sentences that mark topic boundaries)
    for point in segmentation_points:
        print(f"Segmentation point at sentence {point + 1}: {sentences[point]}")

def main():
    """
    Main function to demonstrate usage of the SentenceBERTDetector.
    """
    # Initialize detector
    processor = SubtitleProcessor()

    # Path to the VTT file
    vtt_file = "W1-W6_subtitle_data/1.1.vtt"

    sentences = processor.process_vtt(vtt_file)
    sentences_text = []
    # print("\nSentences found in the VTT file:")
    # print("-" * 80)  # Print a separator line

    for i, sentence in enumerate(sentences, 1):  # enumerate from 1 for readability
        # print(f"\nSentence {i}:")
        # print(f"Start Time: {sentence['start_time']}")
        # print(f"Caption Index: {sentence['index']}")
        # print(f"{sentence['text']}")
        # print("-" * 80)  # Print a separator line between sentences
        sentences_text.append(sentence['text'])

    process_sentence(sentences_text)

if __name__ == "__main__":
    main()