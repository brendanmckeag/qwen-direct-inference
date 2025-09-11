import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import runpod
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
MODEL_NAME = os.environ.get("MODEL_NAME", "microsoft/DialoGPT-medium")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "1000"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
TOP_K = int(os.environ.get("TOP_K", "50"))

class DialoGPTHandler:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model and tokenizer"""
        try:
            logger.info(f"Loading model: {MODEL_NAME}")
            
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            
            # Set pad token for DialoGPT
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
    
    def generate_response(self, prompt: str, conversation_history: list = None, **kwargs) -> str:
        """Generate response from DialoGPT"""
        try:
            # Get generation parameters
            max_length = kwargs.get('max_length', MAX_LENGTH)
            temperature = kwargs.get('temperature', TEMPERATURE)
            top_p = kwargs.get('top_p', TOP_P)
            top_k = kwargs.get('top_k', TOP_K)
            do_sample = kwargs.get('do_sample', True)
            
            # Build conversation context for DialoGPT
            if conversation_history:
                # Include conversation history
                conversation_text = ""
                for turn in conversation_history[-5:]:  # Keep last 5 turns
                    conversation_text += turn + self.tokenizer.eos_token
                conversation_text += prompt + self.tokenizer.eos_token
            else:
                # Single turn conversation
                conversation_text = prompt + self.tokenizer.eos_token
            
            # Tokenize
            inputs = self.tokenizer.encode(
                conversation_text, 
                return_tensors="pt",
                truncate=True,
                max_length=max_length - 100  # Leave room for generation
            ).to(self.device)
            
            input_length = inputs.shape[1]
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=min(input_length + 200, max_length),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3  # Avoid repetition
                )
            
            # Decode only the new tokens
            response = self.tokenizer.decode(
                outputs[0][input_length:], 
                skip_special_tokens=True
            )
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return f"Error during generation: {str(e)}"

# Initialize handler
print("🚀 Initializing DialoGPT handler...")
try:
    handler = DialoGPTHandler()
    print("✅ DialoGPT handler ready!")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    handler = None

def inference(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main inference function"""
    try:
        if handler is None:
            return {"error": "Model not loaded", "status": "error"}
        
        job_input = job.get("input", {})
        prompt = job_input.get("prompt", "")
        conversation_history = job_input.get("conversation_history", [])
        
        if not prompt:
            return {"error": "No prompt provided", "status": "error"}
        
        # Generation parameters
        params = {
            "max_length": job_input.get("max_length", MAX_LENGTH),
            "temperature": job_input.get("temperature", TEMPERATURE),
            "top_p": job_input.get("top_p", TOP_P),
            "top_k": job_input.get("top_k", TOP_K),
            "do_sample": job_input.get("do_sample", True)
        }
        
        response = handler.generate_response(
            prompt, 
            conversation_history=conversation_history,
            **params
        )
        
        return {
            "output": {
                "text": response,
                "model": MODEL_NAME
            },
            "status": "success"
        }
        
    except Exception as e:
        return {"error": str(e), "status": "error"}

if __name__ == "__main__":
    logger.info("Starting RunPod serverless endpoint")
    runpod.serverless.start({"handler": inference})
