import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import AutoConfig, AutoModel
from transformers import AutoProcessor
import time
import warnings

warnings.filterwarnings("ignore")

device = "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=False, use_safetensors=False)
#model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

def whisper (record_file) :
   # start = time.time() 
    result = pipe(record_file)
   # print("Recorded transcript :", result["text"])
   # print("Time duration :" , str(time.time()-start))
    return result["text"]
    
