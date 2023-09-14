from ArabicSpeechRecognition.phonetiseArabic import phonetise
from ArabicSpeechRecognition.speechtotext import get_large_audio_transcription
import ArabicSpeechRecognition.RecordToScore as score
import ArabicSpeechRecognition.WordMatching as wm
import ArabicSpeechRecognition.WordMetrics

def ArSpeechToScore(title):

    real_phoneme = phonetise(title)

    file_path = "hearing.wav"
    transcript_txt = get_large_audio_transcription(file_path)

    recorded_phoneme = phonetise(transcript_txt)


    real_and_transcribed_words, mapped_words_indices = score.matchSampleAndRecordedWords(real_text = title, recorded_transcript = transcript_txt)
    current_words_recorded_accuracy1, current_words_recorded_accuracy2 = score.getPronunciationAccuracy(real_and_transcribed_words)

    real_transcripts = ' '.join(
            [word[0] for word in real_and_transcribed_words])
    matched_transcripts = ' '.join(
            [word[1] for word in real_and_transcribed_words])

                 
    words_real = title.split()
    mapped_words = matched_transcripts.split()

    for idx in range(len(words_real)) :
            if (words_real[idx].find(mapped_words[idx]) != 0) and (words_real[idx].find(mapped_words[idx]) != -1):
                i = words_real[idx].find(mapped_words[idx])
                print ("find1", i)
                while (i>0) :
                     mapped_words[idx] = '_' + mapped_words[idx] 
                     i-=1
            while len(mapped_words[idx]) < len(words_real[idx]):
                mapped_words[idx] = mapped_words[idx] +'_'

    for idx in range(len(mapped_words)) :
            if (mapped_words[idx].find(words_real[idx]) != 0) and (mapped_words[idx].find(words_real[idx]) != -1):
                i = mapped_words[idx].find(words_real[idx])
                print ("find2", i)
                while (i>0) :
                     words_real[idx] = '_' + words_real[idx] 
                     i-=1
            while len(words_real[idx]) < len(mapped_words[idx]):
                words_real[idx] = words_real[idx] +'_'

    is_letter_correct_all_words = ''
    for idx, word_real in enumerate(words_real):

        is_letter_correct = wm.getWhichLettersWereTranscribedCorrectly(word_real, mapped_words[idx])
                    
        is_letter_correct_all_words += ''.join([str(is_correct) for is_correct in is_letter_correct]) + ' '
                    
    binary_txt = is_letter_correct_all_words.split()

    for idx in range(len(mapped_words)) :
        while len(mapped_words[idx]) < len(binary_txt[idx]):
            mapped_words[idx] = mapped_words[idx] +'_'
    
    for idx, word in enumerate(words_real) :
        word = word.replace('_','')
        words_real[idx] = word

    record_txt = ' '.join(
            [word for word in words_real])

    b_txt = ' '.join(
            [word for word in binary_txt])

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
           'result_html' : str(words_result_html)}
    
    return result

