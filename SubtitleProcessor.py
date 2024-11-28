import re           # For regular expressions used in _clean_text()
import webvtt       # For reading VTT files 
                    # Run `pip install webvtt-py` to install
import math
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import numpy as np

class SubtitleProcessor:
   """
   A class to process VTT (WebVTT) files from caption segments into a list of complete sentences.

   After constructing an instance, simply call `.process_vtt(vtt_file)` on the instance to get a list of sentece (a dictionary), where `vtt_file` is the path to the .vtt file.
   Each sentence in the list has:
           - text: str
               The complete sentence text
           - start_time: str
               Start timestamp of the first caption in the sentence
           - index: int
               Index of the first caption in the sentence
   """

   def __init__(self):
       """Initialize the SubtitleProcessor."""
       pass

   def _clean_text(self, text: str) -> str:
       """Clean caption text by removing special markers and normalizing whitespace."""
       # Remove markers like [SOUND], [MUSIC], etc.
       text = re.sub(r'\[.*?\]', '', text)
       text = re.sub(r'>', '', text)
       # Normalize whitespace
       text = ' '.join(text.split())
       return text.strip()

   def _sentence_generator(self, captions: list) -> dict:
       """
       Yields sentences from a sequence of captions.
       Combines consecutive captions until sentence-ending punctuation is found.

       Parameters
       ----------
       captions : list
           List of caption objects from webvtt.read()
           Each caption object should have:
               - text: str
               - start: str (timestamp)

       Yields
       ------
       dict
           Dictionary containing sentence information:
           - text: str
               The complete sentence text
           - start_time: str
               Start timestamp of the first caption in the sentence
           - index: int
               Index of the first caption in the sentence

       Notes
       -----
       - Sentence boundaries are detected by ., !, or ? at the end of text
       - Incomplete sentences at the end of file are still yielded
       """
       current = None

       for i, caption in enumerate(captions):
           cleaned_text = self._clean_text(caption.text)
           if not cleaned_text:
               continue

           # Start a new sentence if we don't have one
           if current is None:
               current = {
                   'text': cleaned_text,
                   'start_time': caption.start,
                   'index': i
               }
           else:
               # Add to existing sentence with space
               current['text'] += f' {cleaned_text}'

           # If we find sentence-ending punctuation, yield the sentence
           if cleaned_text.rstrip().endswith(('.', '!', '?')):
               yield current
               current = None

       # Yield any remaining text as a sentence
       if current:
           yield current

   def process_vtt(self, vtt_file: str) -> list:
       """
       Process a VTT file into a list of sentences using natural sentence boundaries.

       Parameters
       ----------
       vtt_file : str
           Path to the VTT file to process

       Returns
       -------
       list
           List of dictionaries, each containing:
           - text: str
               The complete sentence text
           - start_time: str
               Start timestamp of the first caption in the sentence
           - index: int
               Index of the first caption in the sentence
       """
       return list(self._sentence_generator(webvtt.read(vtt_file)))

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