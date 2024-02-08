import WordMatching as wm
import pronunciationTrainer
import time
import subprocess

def lambda_handler(real_sentence):
    command1 = "/usr/bin/ffmpeg -y -i hearing.wav -ar 16000 -ac 1 hearing2.wav"
    subprocess.run(command1, shell=True)
    command2 = "/usr/bin/mv hearing2.wav hearing.wav"
    subprocess.run(command2, shell=True )
    command3 = "python ~/milo/denoiser/denoiser.py hearing.wav hearing_output.wav"
    subprocess.run(command3,shell=True)

    recorded_audio_file = "hearing_output.wav"
    result = pronunciationTrainer.processAudioForGivenText(recorded_audio_file, real_sentence)

    start = time.time()
    real_transcripts = ' '.join(
        [word[0] for word in result['real_and_transcribed_words']])
    matched_transcripts = ' '.join(
        [word[1] for word in result['real_and_transcribed_words']])

    words_real = real_transcripts.lower().split()
    mapped_words = matched_transcripts.split()

    is_letter_correct_all_words = ''
    for idx, word_real in enumerate(words_real):

        mapped_letters, mapped_letters_indices = wm.get_best_mapped_words(
            mapped_words[idx], word_real)

        is_letter_correct = wm.getWhichLettersWereTranscribedCorrectly(
            word_real, mapped_letters)  # , mapped_letters_indices)

        is_letter_correct_all_words += ''.join([str(is_correct)
                                                for is_correct in is_letter_correct]) + ' '

    print('Time to post-process results: ', str(time.time()-start))

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

    words_result_html = "<div style='text-align:center;'>"
    for char_result in list(zip(record_txt, b_txt)):
        if char_result[1] == '1':
            words_result_html += "<span style= '" + "color:green;font-size:60px;" + " ' >" + char_result[0] + "</span>"
        else:
            words_result_html += "<span style= ' " + "color:red;font-size:60px;" + " ' >" + char_result[0] + "</span>"

    words_result_html += "</div>"

    res = {'recorded_transcript': result['recording_transcript'],
           'pronunciation_accuracy': str(int(result['pronunciation_accuracy'])),
           'real_transcripts': real_transcripts, 
           'is_letter_correct_all_words': is_letter_correct_all_words, 
           'result_html': words_result_html}
    return res
