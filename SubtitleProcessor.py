import re           # For regular expressions used in _clean_text()
import webvtt       # For reading VTT files 
                    # Run `pip install webvtt-py` to install
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

def main():
    """
    Main function to demonstrate usage of the SentenceBERTDetector.
    """
    # Initialize detector
    processor = SubtitleProcessor()

    # Path to the VTT file
    vtt_file = "path to .vtt file"

    sentences = processor.process_vtt(vtt_file)

    print("\nSentences found in the VTT file:")
    print("-" * 80)  # Print a separator line

    for i, sentence in enumerate(sentences, 1):  # enumerate from 1 for readability
        print(f"\nSentence {i}:")
        print(f"Start Time: {sentence['start_time']}")
        print(f"Caption Index: {sentence['index']}")
        print(f"Text: {sentence['text']}")
        print("-" * 80)  # Print a separator line between sentences

if __name__ == "__main__":
    main()