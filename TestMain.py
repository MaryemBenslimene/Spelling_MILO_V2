import lambdaTTS
import lambdaSpeechToScore
import lambdaGetSample
import base64
import io

import base64
import io


def main():

    language = 'en'

    #sentence_to_read = lambdaGetSample.lambda_handler(language)
    #print("Sentence to read : ", sentence_to_read['real_transcript'][0])
    #print("Phonetics : ", sentence_to_read['ipa_transcript'])
    title = 'That isnt how most people do that.'
    #print("Sentence to read : ", sentence_to_read['real_transcript'][0])

    lambdaTTS.lambda_handler(title)

    speech_to_score = lambdaSpeechToScore.lambda_handler(title, language)

    print("Final result:", speech_to_score)



if __name__ == "__main__":
    main()








