from abc import ABC , abstractmethod 
# we are importing the above abc as we need a contract
import torch 

class BaseImageModel(ABC):
    """So the purpose of this class is to make sure that no matter 
    what model is used it could be easily integrated in the pipeline."""
    def __init__(self,model_id:str, device:str = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None 
        self.processor = None 

    @abstractmethod
    def load_model(self):
        """The purpose of this method is to load the model"""
        pass 

    @abstractmethod
    def unload_model(self):
        """The purpose of this method is to remove the model
        from the memory after its work has been done so that it 
        does not consume unnecessary resources """
        pass

    #The below method is being provided with the assumption that the model is going to have the feature of taking the image as the input and is going to generate the output prompt.


    @abstractmethod
    def generate(self, image, task_prompt: str = "<DETAILED_CAPTION>"):
        """
        Vision Logic: Image + Task -> Text/JSON.
        
        :param image: A PIL.Image object or a torch.Tensor (3D: Channels x Height x Width).
        :param task_prompt: The specific Florence-2 task (e.g., '<CAPTION>', '<OD>').
        :return: A dictionary or string containing the model's findings.
        """
        pass
    


    
