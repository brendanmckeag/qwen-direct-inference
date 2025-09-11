import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import runpod
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2048"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
TOP_K = int(os.environ.get("TOP_K", "50"))
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "true").lower() == "true"

class Qwen3Handler:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Qwen3 model and tokenizer"""
        try:
            logger.info(f"Loading model: {MODEL_NAME}")
            
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            
            # Model loading configuration
            model_kwargs = {
                "torch_dtype": "auto",
                "device_map": "auto",
                "low_cpu_mem_usage": True
            }
            
            # Use flash-attention if available
            if FLASH_ATTN_AVAILABLE:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using flash-attention for faster inference")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                **model_kwargs
            )
            
            logger.info("Qwen3 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
    
    def _format_conversation(self, prompt: str, conversation_history: List[Dict] = None) -> List[Dict]:
        """Format conversation for Qwen3 chat template"""
        messages = []
        
        # Add conversation history if provided
        if conversation_history:
            # Ensure conversation history is in the correct format
            for entry in conversation_history[-10:]:  # Keep last 10 turns
                if isinstance(entry, dict) and "role" in entry and "content" in entry:
                    messages.append(entry)
                elif isinstance(entry, str):
                    # Convert string to message format (assume alternating user/assistant)
                    role = "assistant" if len(messages) % 2 == 1 else "user"
                    messages.append({"role": role, "content": entry})
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def _parse_qwen3_output(self, output_ids: List[int]) -> Dict[str, str]:
        """Parse Qwen3 output to separate thinking and content"""
        try:
            # Find the </think> token (151668) from the end
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            # No thinking content found
            index = 0
        
        thinking_content = ""
        content = ""
        
        if index > 0:
            thinking_content = self.tokenizer.decode(
                output_ids[:index], 
                skip_special_tokens=True
            ).strip()
        
        content = self.tokenizer.decode(
            output_ids[index:], 
            skip_special_tokens=True
        ).strip()
        
        return {
            "thinking": thinking_content,
            "content": content
        }
    
    def generate_response(self, prompt: str, conversation_history: List[Dict] = None, **kwargs) -> Dict[str, str]:
        """Generate response from Qwen3"""
        try:
            # Get generation parameters
            max_new_tokens = kwargs.get('max_new_tokens', MAX_NEW_TOKENS)
            temperature = kwargs.get('temperature', TEMPERATURE)
            top_p = kwargs.get('top_p', TOP_P)
            top_k = kwargs.get('top_k', TOP_K)
            do_sample = kwargs.get('do_sample', True)
            enable_thinking = kwargs.get('enable_thinking', ENABLE_THINKING)
            
            # Format conversation
            messages = self._format_conversation(prompt, conversation_history)
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
            
            # Tokenize
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Extract only the new tokens
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            
            # Parse thinking and content
            parsed_output = self._parse_qwen3_output(output_ids)
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return parsed_output
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                "thinking": "",
                "content": f"Error during generation: {str(e)}"
            }

# Initialize handler
print("🚀 Initializing Qwen3 handler...")
try:
    handler = Qwen3Handler()
    print("✅ Qwen3 handler ready!")
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
            "max_new_tokens": job_input.get("max_new_tokens", MAX_NEW_TOKENS),
            "temperature": job_input.get("temperature", TEMPERATURE),
            "top_p": job_input.get("top_p", TOP_P),
            "top_k": job_input.get("top_k", TOP_K),
            "do_sample": job_input.get("do_sample", True),
            "enable_thinking": job_input.get("enable_thinking", ENABLE_THINKING)
        }
        
        response = handler.generate_response(
            prompt, 
            conversation_history=conversation_history,
            **params
        )
        
        return {
            "output": {
                "text": response["content"],
                "thinking": response["thinking"],
                "model": MODEL_NAME,
                "thinking_enabled": params["enable_thinking"]
            },
            "status": "success"
        }
        
    except Exception as e:
        return {"error": str(e), "status": "error"}

if __name__ == "__main__":
    logger.info("Starting RunPod serverless endpoint with Qwen3")
    runpod.serverless.start({"handler": inference})
