import ModelInterfaces
import torch
import numpy as np
from sileromodels.src.silero import silero_tts
from sileromodels.src.silero.utils import prepare_model_input


class NeuralASR(ModelInterfaces.IASRModel):
    word_locations_in_samples = None
    audio_transcript = None

    def __init__(self, model: torch.nn.Module, decoder) -> None:
        super().__init__()
        self.model = model
        self.decoder = decoder  # Decoder from CTC-outputs to transcripts

    def getTranscript(self) -> str:
        """Get the transcripts of the process audio"""
        assert(self.audio_transcript != None,
               'Can get audio transcripts without having processed the audio')
        return self.audio_transcript

    def getWordLocations(self) -> list:
        """Get the pair of words location from audio"""
        assert(self.word_locations_in_samples != None,
               'Can get word locations without having processed the audio')

        return self.word_locations_in_samples

    def processAudio(self, audio: torch.Tensor):
        """Process the audio"""
        audio_length_in_samples = audio.shape[1]
        audio = prepare_model_input(audio, device = torch.device('cpu'))
        with torch.inference_mode():
            nn_output = self.model(audio)

            self.audio_transcript, self.word_locations_in_samples = self.decoder(
                nn_output[0, :, :].detach(), audio_length_in_samples, word_align=True)
            print("self.audio_transcript", self.audio_transcript)
            print("self.word_locations_in_samples", self.word_locations_in_samples)


class NeuralTTS(ModelInterfaces.ITextToSpeechModel):
    def __init__(self, model: torch.nn.Module, sampling_rate: int) -> None:
        super().__init__()
        self.model = model
        self.sampling_rate = sampling_rate

    def getAudioFromSentence(self, sentence: str) -> np.array:
        with torch.inference_mode():
            model, symbols, sample_rate, example_text, apply_tts = silero_tts(language='en', speaker = 'lj_16khz')
            audio_transcript = apply_tts(texts=[sentence], model = model, symbols = symbols, sample_rate=self.sampling_rate, device = torch.device('cpu'))[0]

        return audio_transcript


class NeuralTranslator(ModelInterfaces.ITranslationModel):
    def __init__(self, model: torch.nn.Module, tokenizer) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def translateSentence(self, sentence: str) -> str:
        """Get the transcripts of the process audio"""
        tokenized_text = self.tokenizer(sentence, return_tensors='pt')
        translation = self.model.generate(**tokenized_text)
        translated_text = self.tokenizer.batch_decode(
            translation, skip_special_tokens=True)[0]

        return translated_text
