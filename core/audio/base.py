from abc import ABC , abstractmethod 
# we are importing the above abc as we need a contract
import torch 

class BaseAudioModel(ABC):
    """So the purpose of this class is to make sure that no matter 
    what model is used it could be easily integrated in the pipeline."""
    def __init__(self,model_id:str, device:str = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None 
        self.processor = None 
    
