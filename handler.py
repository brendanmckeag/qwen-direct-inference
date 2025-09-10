import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import runpod
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-1.7B")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "4096"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
TOP_K = int(os.environ.get("TOP_K", "50"))

class QwenInferenceHandler:
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
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True
            )
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with basic settings
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from the model"""
        try:
            # Get generation parameters
            max_length = kwargs.get('max_length', MAX_LENGTH)
            temperature = kwargs.get('temperature', TEMPERATURE)
            top_p = kwargs.get('top_p', TOP_P)
            top_k = kwargs.get('top_k', TOP_K)
            do_sample = kwargs.get('do_sample', True)
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            input_length = inputs.shape[1]
            
            # Calculate max new tokens
            max_new_tokens = min(max_length - input_length, 1024)
            if max_new_tokens <= 0:
                return "Error: Input prompt is too long"
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
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
print("🚀 Initializing model handler...")
try:
    handler = QwenInferenceHandler()
    print("✅ Model handler ready!")
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
        
        response = handler.generate_response(prompt, **params)
        
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
