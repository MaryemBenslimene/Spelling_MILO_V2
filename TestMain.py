
import lambdaTTS
import lambdaSpeechToScore
import lambdaGetSample
from Arabic_SpeechToScore import ArSpeechToScore

import base64
import io
import base64
import io
from flask import Flask
from flask import request
from flask import Response
import json
import base64

application = Flask(__name__) 
@application.route('/model/spelling', methods=['POST'])
def execute_spelling():
    try :
        language =  request.get_json().get("language")
        encode_string = request.get_json().get("voice")
        title = request.get_json().get("title")
        wav_file = open("hearing.wav", "wb")
        decode_string = base64.b64decode(encode_string)
        wav_file.write(decode_string)

        if language == 'en' :
            lambdaTTS.lambda_handler(title)
            speech_to_score = lambdaSpeechToScore.lambda_handler(title, language)
            return Response(json.dumps({'result': speech_to_score }),
                            status=200,
                            mimetype="application/json")
        
        elif language == 'ar' :
            result = ArSpeechToScore(title)
            return Response(json.dumps({'result': result }),
                            status=200,
                            mimetype="application/json")
    except :
        speech_to_score = {
            "recorded_transcript": "",
            "ipa_recorded_transcript": "",
            "pronunciation_accuracy": "",
            "real_transcripts": "",
            "real_transcripts_ipa": "",
            "is_letter_correct_all_words": "",
            "result_html": "<span style= '" + "color:red" + " ' > Error ! Please try again. </span>"}
        return Response(json.dumps({'result': speech_to_score }),status=200)

	
if __name__ == '__main__':
    application.run(host='0.0.0.0', port=3000,debug=False)

