import torchaudio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2Config
)
import torch
import epitran
import WordMatching as wm 
import WordMetrics
import lambdaSpeechToScore as score
import subprocess
import soundfile as sf


model_name = "facebook/wav2vec2-large-xlsr-53-french"
device = "cpu"

model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
processor = Wav2Vec2Processor.from_pretrained(model_name)

resampler = torchaudio.transforms.Resample(orig_freq=48_000, new_freq=16_000)

def ASR_french () :
    #command = "python ~/milo/denoiser/denoiser.py hearing.wav hearing_output.wav"

    #result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    command1 = "/usr/bin/ffmpeg -y -i hearing.wav -ar 16000 -ac 1 hearing2.wav"
    #command1 = "ffmpeg -y -i hearing.wav -ar 16000 -ac 1 hearing2.wav"
    subprocess.run(command1, shell=True)
    command2 = "/usr/bin/mv hearing2.wav hearing.wav"
    subprocess.run(command2, shell=True )
    command3 = "python ~/milo/denoiser/denoiser.py hearing.wav hearing_output.wav"
    subprocess.run(command3,shell=True)

    recorded_audio_file = "hearing_output.wav"
  
    audio_data, sample_rate = sf.read(recorded_audio_file)

    features = processor(audio_data, sampling_rate=sample_rate , padding=True, return_tensors="pt")
    input_values = features.input_values.to(device)
   
    attention_mask = features.attention_mask.to(device)
    with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits
    pred_ids = torch.argmax(logits, dim=-1)
    pred = processor.batch_decode(pred_ids)
    
    return(pred[0])

def phonemizer_fr(sentence) :
    phonem_converter = epitran.Epitran('fra-Latn')
    recording_ipa = phonem_converter.transliterate(sentence)
    return (recording_ipa)

    
def FrSpeechToScore(title) :
    real_phonemes = phonemizer_fr(title)

    transcript_txt = ASR_french()
    recorded_phonemes = phonemizer_fr(transcript_txt)
  
    title = title.lower()
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

    words_result_html = "<div style='text-align:center;'>"
    for char_result in list(zip(record_txt, b_txt)):
        if char_result[1] == '1':
            words_result_html += "<span style= '" + "color:green;font-size:60px;" + " ' >" + char_result[0] + "</span>"
        else:
            words_result_html += "<span style= ' " + "color:red;font-size:60px;" + " ' >" + char_result[0] + "</span>"

    words_result_html += "</div>"
    result = {'recorded_transcript': transcript_txt,
           'ipa_recorded_transcript': recorded_phonemes,
           'pronunciation_accuracy': str(current_words_recorded_accuracy1),
           'real_transcripts': title, 
           'real_transcripts_ipa': real_phonemes, 
           'is_letter_correct_all_words': is_letter_correct_all_words,
           'result_html' : str(words_result_html)}
    
    return result 
