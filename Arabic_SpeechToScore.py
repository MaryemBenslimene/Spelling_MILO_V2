from ArabicSpeechRecognition.phonetiseArabic import phonetise
from ArabicSpeechRecognition.speechtotext import get_large_audio_transcription
import ArabicSpeechRecognition.RecordToScore as score
import ArabicSpeechRecognition.WordMatching as wm
import ArabicSpeechRecognition.WordMetrics


def ArSpeechToScore(title):

    real_phoneme = phonetise(title)

    print("Real Phonemes", real_phoneme)
    file_path = "/home/ubuntu/milo/Spelling_trainer/hearing.wav"
    transcript_txt = get_large_audio_transcription(file_path)

    recorded_phoneme = phonetise(transcript_txt)

    print("Recorded Phonemes :", recorded_phoneme)

    real_and_transcribed_words, mapped_words_indices = score.matchSampleAndRecordedWords(real_text = title, recorded_transcript = transcript_txt)
    print("real_and_transcribed_words", real_and_transcribed_words)
    current_words_recorded_accuracy1, current_words_recorded_accuracy2 = score.getPronunciationAccuracy(real_and_transcribed_words)

    print("current_words_handwritten_accuracy1", current_words_recorded_accuracy1)
    print("current_words_handwritten_accuracy2", current_words_recorded_accuracy2)


    real_transcripts = ' '.join(
            [word[0] for word in real_and_transcribed_words])
    matched_transcripts = ' '.join(
            [word[1] for word in real_and_transcribed_words])

                 
    words_real = real_transcripts.split()
    mapped_words = matched_transcripts.split()

    is_letter_correct_all_words = ''
    for idx, word_real in enumerate(words_real):

        is_letter_correct = wm.getWhichLettersWereTranscribedCorrectly(word_real, mapped_words[idx])
                    
        is_letter_correct_all_words += ''.join([str(is_correct) for is_correct in is_letter_correct]) + ' '
                    
    print("is_letter_correct_all_words", is_letter_correct_all_words)

    binary_txt = is_letter_correct_all_words.split()
    print("handwritten_txt", mapped_words)
    print("binary_txt", binary_txt)
    for idx in range(len(mapped_words)) :
        while len(mapped_words[idx]) < len(binary_txt[idx]):
            mapped_words[idx] = mapped_words[idx] +'_'


    print("Recorded_txt", mapped_words)
    print("Binary_txt", binary_txt)

    record_txt = ' '.join(
            [word for word in mapped_words])

    b_txt = ' '.join(
            [word for word in binary_txt])

    print("hand_txt", mapped_words)
    print("b_txt", b_txt)
    words_result_html = "<div>"
    for char_result in list(zip(record_txt, b_txt)):
        if char_result[1] == '1':
            words_result_html += "<span style= '" + "color:green" + " ' >" + char_result[0] + "</span>"
        else:
            words_result_html += "<span style= ' " + "color:red" + " ' >" + char_result[0] + "</span>"

    words_result_html += "</div>"

    result = {'recorded_transcript': transcript_txt,
           'ipa_recorded_transcript': recorded_phoneme,
           'pronunciation_accuracy': str(current_words_recorded_accuracy1),
           'real_transcripts': title, 
           'real_transcripts_ipa': real_phoneme, 
           'is_letter_correct_all_words': is_letter_correct_all_words,
           'result_html' : words_result_html,
           'words_real' : words_real,
           'mapped_words' : mapped_words}
    
    return result
