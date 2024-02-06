
import lambdaSpeechToScore
from Arabic_SpeechToScore import ArSpeechToScore
from French_SpeechToScore import FrSpeechToScore

import base64
import io
import base64
import io
from flask import Flask
from flask import request
from flask import Response
import json
import base64
import traceback

application = Flask(__name__) 
@application.route('/model/spelling', methods=['POST'])
def execute_spelling():
    try :
        #traceback.print_exc()
        language =  request.get_json().get("language")
        encode_string = request.get_json().get("voice")
        title = request.get_json().get("title")
        print(language, title)
        wav_file = open("hearing.wav", "wb")
        decode_string = base64.b64decode(encode_string)
        wav_file.write(decode_string)

        if language == 'en' :
            print("in language en")
            speech_to_score = lambdaSpeechToScore.lambda_handler(title, language)
            #traceback.print_exc()
            return Response(json.dumps({'result': speech_to_score }),
                            status=200,
                            mimetype="application/json")

        elif language == 'ar' :
            print("in language ar")
            result = ArSpeechToScore(title)
            #traceback.print_exc()
            return Response(json.dumps({'result': result }),
                            status=200,
                            mimetype="application/json")

        elif language == 'fr' :
            result = FrSpeechToScore(title)
            return Response(json.dumps({'result': result }),
                            status=200,
                            mimetype="application/json")

    except :
        traceback.print_exc()
        speech_to_score = {
            "recorded_transcript": "",
            "ipa_recorded_transcript": "",
            "pronunciation_accuracy": "",
            "real_transcripts": "",
            "real_transcripts_ipa": "",
            "is_letter_correct_all_words": "",
            "result_html": "<span style= '" + "color:red;font-size:20px;" + " ' > Error ! Please try again. </span>"}
        return Response(json.dumps({'result': speech_to_score }),status=200)

	
if __name__ == '__main__':
    application.run(host='0.0.0.0', port=3000,debug=False)

