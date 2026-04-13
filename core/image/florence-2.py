# import torch
# from .base import BaseImageModel
# from PIL import Image
# from transformers import AutoProcessor, AutoModelForCausalLM, AutoImageProcessor, AutoTokenizer
# from typing import override


# class Florence_2Model(BaseImageModel):

#     def __init__(self, model_id = "microsoft/Florence-2-base", device = None):
#         super().__init__(model_id, device)
#         self.image_processor = None
#         self.tokenizer = None


#     @override
#     def load_model(self):
#         """ This is the manual loading of the model so that the resources could be 
#         easily handled."""
#         try:
#             print("Attempting to load processor...")
#             self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
#         except AttributeError as e:
#             if "forced_bos_token_id" in str(e):
#                 print("⚠️  Detected Florence-2 configuration bug, using workaround...")
#                 try:
#                     # Try loading tokenizer with trust_remote_code=False to avoid config
#                     self.processor = None
#                     print("Loading image processor...")
#                     self.image_processor = AutoImageProcessor.from_pretrained(self.model_id, trust_remote_code=True)
#                     print("Loading tokenizer (this may take a moment)...")
#                     # Load tokenizer without trust_remote_code to bypass the buggy config
#                     self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
#                 except AttributeError as e2:
#                     print(f"⚠️  Tokenizer loading also failed: {e2}")
#                     print("Attempting final fallback: loading only image processor...")
#                     self.processor = None
#                     self.image_processor = AutoImageProcessor.from_pretrained(self.model_id, trust_remote_code=True)
#                     self.tokenizer = None
#             else:
#                 raise
        
#         print("Loading model weights (this may take several minutes)...")
#         self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, torch_dtype=torch.float32 if self.device == "cpu" else torch.float16, low_cpu_mem_usage=True,attn_implementation="eager").to(self.device)

#         self.model.eval()
#         print("✓ Model loaded successfully")




#     @override
#     # def generate(self, image , task_prompt:str = "<DETAILED_CAPTION>"):
#     #     """ This method is going to generate the output of the image that we are going to pass"""

#     #     # Handle both processor and separate image_processor/tokenizer cases
#     #     if self.processor is not None:
#     #         inputs = self.processor(image=image, text=task_prompt, return_tensors="pt").to(self.device)
#     #     else:
#     #         # Fallback if processor has issues - manually combine
#     #         image_inputs = self.image_processor(image, return_tensors="pt")
#     #         text_inputs = self.tokenizer(task_prompt, return_tensors="pt")
#     #         inputs = {**image_inputs, **text_inputs}
#     #         for key in inputs:
#     #             inputs[key] = inputs[key].to(self.device)

#     #     with torch.no_grad():
#     #         generated_ids = self.model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=512, do_sample=False, num_beams=3)

#     #     generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

#     #     # Parse the output
#     #     if self.processor is not None:
#     #         parsed_answer = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.height, image.width))
#     #     else:
#     #         parsed_answer = {task_prompt: generated_text}

#     #     return parsed_answer[task_prompt]

#     def generate(self, image, task_prompt: str = "<DETAILED_CAPTION>"):
#     # Analyzing the fix: We change 'image=' to 'images=' to match the Florence2Processor signature.
#     # We also ensure the task_prompt is correctly passed.
    
#     # Check if we are using the combined processor or the decoupled workaround
#         if self.processor:
#             inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)
#         else:
#             # Fallback logic you documented in DEBUGGING_PROMPTS
#             pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values
#             input_ids = self.tokenizer(text=task_prompt, return_tensors="pt").input_ids
#             inputs = {
#                 "pixel_values": pixel_values.to(self.device), 
#                 "input_ids": input_ids.to(self.device)
#             }

#     # The actual inference step
#         generated_ids = self.model.generate(
#             input_ids=inputs["input_ids"],
#             pixel_values=inputs["pixel_values"],
#             max_new_tokens=1024,
#             num_beams=1,
#             do_sample=False
#         )

#     # Post-processing the output
#         generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
#         parsed_answer = self.processor.post_process_generation(
#             generated_text, 
#             task=task_prompt, 
#             image_size=(image.width, image.height)
#         )

#         return parsed_answer
    

#     @override 
#     def unload_model(self):
#         """This is the method to unload the model"""

#         import gc
#         del self.model 
#         if hasattr(self, 'processor') and self.processor is not None:
#             del self.processor
#         if hasattr(self, 'image_processor'):
#             del self.image_processor
#         if hasattr(self, 'tokenizer'):
#             del self.tokenizer
#         self.model = None
#         self.processor = None 
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         print("Model unloaded successfully")

import torch
import time
from .base import BaseImageModel
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoImageProcessor, AutoTokenizer
from typing import override

class Florence_2Model(BaseImageModel):

    def __init__(self, model_id="microsoft/Florence-2-base", device=None):
        super().__init__(model_id, device)
        self.image_processor = None
        self.tokenizer = None
        self.processor = None

    @override
    def load_model(self):
        """Manual loading with local-first priority and bug workarounds."""
        try:
            print("Attempting to load processor...")
            # Using local_files_only=True if you've already downloaded successfully
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, 
                trust_remote_code=True,
                local_files_only=True
            )
        except Exception as e:
            print(f"⚠️ Standard loader failed: {e}. Using decoupled fallback...")
            self.processor = None
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_id, trust_remote_code=True,local_files_only=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id,local_files_only=True)
        
        print("Loading model weights (Baseline FP32)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            trust_remote_code=True, 
            local_files_only=True,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16, 
            low_cpu_mem_usage=True,
            attn_implementation="eager" # Avoids the SDPA attribute error
        ).to(self.device)

        self.model.eval()
        print("✓ Model loaded successfully")

    @override
    def generate(self, image, task_prompt: str = "<DETAILED_CAPTION>"):
        """Generates the caption with early stopping and no_grad for memory efficiency."""
        
        print("DEBUG: Preparing inputs...")
        if self.processor:
            # Plural 'images' is required by the Florence2Processor signature
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)
        else:
            pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values
            input_ids = self.tokenizer(text=task_prompt, return_tensors="pt").input_ids
            inputs = {
                "pixel_values": pixel_values.to(self.device), 
                "input_ids": input_ids.to(self.device)
            }

        print(f"DEBUG: Starting inference on {self.device} (Greedy Search)...")
        start_time = time.time()

        # Disabling gradient calculation reduces memory consumption significantly
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=1,           # Baseline optimization
                do_sample=False,       # Greedy search for reproducibility
                early_stopping=True    # Prevent CPU from wasting cycles on padding
            )

        print(f"DEBUG: Inference complete in {time.time() - start_time:.2f}s. Decoding...")

        # Select the correct tokenizer for decoding
        tokenizer_to_use = self.processor.tokenizer if self.processor else self.tokenizer
        generated_text = tokenizer_to_use.batch_decode(generated_ids, skip_special_tokens=False)[0]

        if self.processor:
            parsed_answer = self.processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(image.width, image.height)
            )
        else:
            parsed_answer = {task_prompt: generated_text}

        return parsed_answer

    @override 
    def unload_model(self):
        """Unloads model and triggers garbage collection to free RAM."""
        import gc
        del self.model 
        if self.processor: del self.processor
        if self.image_processor: del self.image_processor
        if self.tokenizer: del self.tokenizer
        
        self.model = None
        self.processor = None 
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Model unloaded successfully. RAM reclaimed.")
        











       











