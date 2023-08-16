
import models
import soundfile as sf
import json
import AIModels
#from flask import Response
import utilsFileIO
import os
import base64
import time

sampling_rate = 16000
model_TTS_lambda = AIModels.NeuralTTS(models.getTTSModel('en'), sampling_rate)


def lambda_handler(sentence):

    linear_factor = 0.2
    audio = model_TTS_lambda.getAudioFromSentence(sentence).detach().numpy()*linear_factor

    #random_file_name = utilsFileIO.generateRandomString(20)+'.wav'

    TTS_audio_file_name = 'TTS_audio_file_name.wav'

    sf.write('./'+TTS_audio_file_name, audio, 16000)

    with open(TTS_audio_file_name, "rb") as f:
        audio_byte_array = f.read()

    #os.remove(random_file_name)
    file_path = "base64Audio.txt"

    with open(file_path, "w") as file:
        file.write("data:audio/ogg;;base64," + str(base64.b64encode(audio_byte_array))[2:-1])

    with open("base64Audio.txt", "r") as file:
        base64Audio_str = file.read()

    language = 'en'

    file_bytes = base64.b64decode(base64Audio_str[22:].encode('utf-8'))

    start = time.time()
    f = open(TTS_audio_file_name, 'wb')
    f.write(file_bytes)
    f.close()

    print('Time for saving binary in file: ', str(time.time()-start))