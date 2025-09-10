import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import runpod
from typing import Dict, Any
import gc
import logging
# test
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-32B")
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
                trust_remote_code=True,
                use_fast=False
            )
            
            # Load model with optimizations for large models
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for better B200 performance
                device_map="auto",  # Automatically distribute across available GPUs
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
            )
            
            # Enable compilation for better performance on newer GPUs
            if hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model, mode="max-autotune")
                    logger.info("Model compilation enabled")
                except Exception as e:
                    logger.warning(f"Model compilation failed: {e}")
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from the model"""
        try:
            # Get generation parameters with defaults
            max_length = kwargs.get('max_length', MAX_LENGTH)
            temperature = kwargs.get('temperature', TEMPERATURE)
            top_p = kwargs.get('top_p', TOP_P)
            top_k = kwargs.get('top_k', TOP_K)
            do_sample = kwargs.get('do_sample', True)
            pad_token_id = kwargs.get('pad_token_id', self.tokenizer.eos_token_id)
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=kwargs.get('repetition_penalty', 1.1),
                    length_penalty=kwargs.get('length_penalty', 1.0),
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            # Clear CUDA cache to prevent memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise e

# =============================================================================
# GLOBAL MODEL INITIALIZATION
# =============================================================================
# We create ONE instance of our model handler when the serverless function starts.
# This is important because loading a 32B model takes time and memory.
# By doing this once at startup, all subsequent requests can reuse the loaded model.
print("🚀 Initializing model handler (this happens once at startup)...")
handler = QwenInferenceHandler()
print("✅ Model handler ready!")

# =============================================================================
# MAIN SERVERLESS FUNCTION: This is called for each inference request
# =============================================================================
def inference(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    This is the main function that RunPod calls for each inference request.
    It receives a job (request) and returns a response.
    
    Args:
        job: A dictionary containing the request data from the user
        
    Returns:
        A dictionary containing either the generated text or an error message
        
    Expected input format from user:
    {
        "input": {
            "prompt": "Your question or prompt here",
            "max_length": 2048,        # Optional: max tokens to generate
            "temperature": 0.7,        # Optional: randomness level
            "top_p": 0.9,             # Optional: nucleus sampling
            "top_k": 50,              # Optional: top-k sampling
            "do_sample": true,        # Optional: use sampling vs greedy
            "repetition_penalty": 1.1, # Optional: prevent repetition
            "length_penalty": 1.0     # Optional: length preference
        }
    }
    """
    try:
        # =================================================================
        # STEP 1: Extract and validate the input data
        # =================================================================
        # Get the input data from the job, default to empty dict if missing
        job_input = job.get("input", {})
        
        # Check that the user provided a prompt (this is required!)
        if "prompt" not in job_input:
            return {
                "error": "❌ Missing required parameter: 'prompt'. Please provide a text prompt to generate from.",
                "status": "error"
            }
        
        prompt = job_input["prompt"]
        
        # =================================================================
        # STEP 2: Prepare generation parameters
        # =================================================================
        # Extract optional parameters, using defaults if not provided
        generation_params = {
            "max_length": job_input.get("max_length", MAX_LENGTH),
            "temperature": job_input.get("temperature", TEMPERATURE),
            "top_p": job_input.get("top_p", TOP_P),
            "top_k": job_input.get("top_k", TOP_K),
            "do_sample": job_input.get("do_sample", True),
            "repetition_penalty": job_input.get("repetition_penalty", 1.1),
            "length_penalty": job_input.get("length_penalty", 1.0),
        }
        
        logger.info(f"🔄 Processing inference request with prompt length: {len(prompt)} characters")
        
        # =================================================================
        # STEP 3: Generate the response using our model
        # =================================================================
        response = handler.generate_response(prompt, **generation_params)
        
        # =================================================================
        # STEP 4: Return successful response
        # =================================================================
        return {
            "output": {
                "text": response,                    # The generated text
                "model": MODEL_NAME,                 # Which model was used
                "parameters": generation_params      # What settings were used
            },
            "status": "success"
        }
        
    except Exception as e:
        # =================================================================
        # ERROR HANDLING: If anything goes wrong, return an error response
        # =================================================================
        logger.error(f"❌ Inference error: {e}")
        return {
            "error": str(e),
            "status": "error"
        }

# =============================================================================
# HEALTH CHECK FUNCTION: Verifies everything is working correctly
# =============================================================================
def health_check():
    """
    This function checks if the serverless endpoint is healthy and ready to serve requests.
    RunPod can call this to verify the endpoint is working properly.
    
    Returns:
        Dictionary with health status and basic info
    """
    try:
        # Check if our model and tokenizer loaded properly
        if handler.model is None or handler.tokenizer is None:
            return {
                "status": "unhealthy", 
                "message": "❌ Model or tokenizer not loaded properly"
            }
        
        # Do a quick test generation to make sure everything works
        # We use a very short prompt and limit generation to save time/resources
        test_response = handler.generate_response(
            "Hello",           # Simple test prompt
            max_length=10,     # Only generate a few tokens
            do_sample=False    # Use greedy decoding for consistent results
        )
        
        # If we got here, everything is working!
        return {
            "status": "healthy", 
            "message": "✅ Endpoint is ready for inference",
            "model": MODEL_NAME,
            "device": str(handler.device),
            "test_generation": test_response  # Include test output for verification
        }
        
    except Exception as e:
        # If the health check fails, return unhealthy status
        return {
            "status": "unhealthy", 
            "error": f"❌ Health check failed: {str(e)}"
        }

# =============================================================================
# MAIN ENTRY POINT: This runs when the serverless function starts
# =============================================================================
if __name__ == "__main__":
    logger.info("🚀 Starting RunPod serverless endpoint for Qwen inference")
    logger.info(f"📦 Model: {MODEL_NAME}")
    logger.info(f"💻 Device: {handler.device}")
    logger.info(f"🔧 Max Length: {MAX_LENGTH}, Temperature: {TEMPERATURE}, Top-P: {TOP_P}, Top-K: {TOP_K}")
    
    # Start the serverless endpoint with our inference function
    # return_aggregate_stream=True enables streaming responses (not used here but good to have)
    runpod.serverless.start({
        "handler": inference,              # The main function to call for each request
        "return_aggregate_stream": True    # Enable streaming capability
    })
    
    logger.info("🎉 Serverless endpoint is now running and ready for requests!")
