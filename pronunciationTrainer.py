import numpy as np
import WordMetrics
import WordMatching as wm
import whisper
from string import punctuation
import time

def processAudioForGivenText(recorded_audio_file=None, real_text=None):

    start = time.time()
    recording_transcript = getAudioTranscript(recorded_audio_file)
        
    print('Time for NN to transcript audio: ', str(time.time()-start))
        
    start = time.time()
    real_and_transcribed_words, mapped_words_indices = matchSampleAndRecordedWords(real_text, recording_transcript)
    print('Time for matching transcripts: ', str(time.time()-start))

    pronunciation_accuracy = getPronunciationAccuracy(real_and_transcribed_words)  # _ipa

    result = {'recording_transcript': recording_transcript,
                'real_and_transcribed_words': real_and_transcribed_words,
                'pronunciation_accuracy': pronunciation_accuracy}

    return result

def getAudioTranscript(RecordedAudioFile):
   
    current_recorded_transcript = whisper.whisper(RecordedAudioFile)
    current_recorded_transcript = " "
    
    return current_recorded_transcript


def matchSampleAndRecordedWords(real_text, recorded_transcript):
    words_estimated = recorded_transcript.split()

    if real_text is None:
        words_real = recorded_transcript[0].split()
    else:
        words_real = real_text.split()

    mapped_words, mapped_words_indices = wm.get_best_mapped_words(words_estimated, words_real)

    real_and_transcribed_words = []
    for word_idx in range(len(words_real)):
        if word_idx >= len(mapped_words)-1:
            mapped_words.append('-')
        real_and_transcribed_words.append(
            (words_real[word_idx], mapped_words[word_idx]))

    return real_and_transcribed_words, mapped_words_indices

def getPronunciationAccuracy(real_and_transcribed_words) -> float:
    total_mismatches = 0.
    number_of_char = 0.
    #current_words_pronunciation_accuracy = []
    for pair in real_and_transcribed_words:
        real_without_punctuation = removePunctuation(pair[0]).lower()
        number_of_word_mismatches = WordMetrics.edit_distance_python(
            real_without_punctuation, removePunctuation(pair[1]).lower())
        total_mismatches += number_of_word_mismatches
        number_of_char_in_word = len(real_without_punctuation)
        number_of_char += number_of_char_in_word

    #    current_words_pronunciation_accuracy.append(float(
    #        number_of_char_in_word-number_of_word_mismatches)/number_of_char_in_word*100)

    percentage_of_correct_pronunciations = (
        number_of_char-total_mismatches)/number_of_char*100

    return np.round(percentage_of_correct_pronunciations)

def removePunctuation(word: str) -> str:
    return ''.join([char for char in word if char not in punctuation])

"""
    def preprocessAudio(self, audio: torch.tensor) -> torch.tensor:
        audio = audio-torch.mean(audio)
        audio = audio/torch.max(torch.abs(audio))
        return audio

"""

