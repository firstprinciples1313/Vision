# # import os
# # import sys
# # import torch
# # import importlib.util
# # from PIL import Image

# # # 1. Add the root 'Vision' folder to sys.path
# # # This allows the florence-2.py file to find its sibling 'base.py'
# # root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# # sys.path.append(root_path)

# # # 2. Manually load the module because of the hyphen in 'florence-2.py'
# # # Python's 'import' keyword hates hyphens, so we use the loader:
# # module_path = os.path.join(root_path, "core", "image", "florence-2.py")
# # spec = importlib.util.spec_from_file_location("florence_module", module_path)
# # florence_module = importlib.util.module_from_spec(spec)
# # spec.loader.exec_module(florence_module)

# import os
# import sys
# import importlib.util

# # 1. Go up 3 levels from 'experiments/image_model/model/' to reach 'Vision/'
# # __file__ is experiments/image_model/model/test_florence-2.py
# current_dir = os.path.dirname(os.path.abspath(__file__))
# root_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))

# # 2. Add the root 'Vision' folder to sys.path
# sys.path.append(root_path)

# # 3. Correct the module path
# module_path = os.path.join(root_path, "core", "image", "florence-2.py")

# # 4. Load the module
# spec = importlib.util.spec_from_file_location("florence_module", module_path)
# florence_module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(florence_module)

# def run_test():
#     # 3. Access the class from the manually loaded module
#     # We'll use CPU to stay safe on 16GB RAM for the first run
#     image_model = florence_module.Florence_2Model(device="cpu")

#     image_model.load_model()

#     # 4. Image handling
#     image_path = os.path.join(os.path.dirname(__file__), "image1.png")
#     if not os.path.exists(image_path):
#         print(f"Error: Put 'image1.png' in {os.path.dirname(__file__)}")
#         return

#     image = Image.open(image_path).convert("RGB")

#     # 5. Execute
#     print("Generating...")
#     result = image_model.generate(image)
    
#     print(f"\nResult: {result}\n")

#     image_model.unload_model()

# if __name__ == "__main__":
#     run_test()


import os
import sys
import torch
import importlib.util
from PIL import Image

# 1. Dynamically find the project root (Vision/)
# We go up 3 levels: model -> image_model -> experiments -> Vision
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))

# 2. Add the root to sys.path so 'core' can be found
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# 3. Use the importlib "Package" trick
# This tells Python: "Load this file, but treat it as if it's inside the core.image package"
module_name = "core.image.florence_2" # We use an underscore for the alias
file_path = os.path.join(root_path, "core", "image", "florence-2.py")

spec = importlib.util.spec_from_file_location(module_name, file_path)
florence_module = importlib.util.module_from_spec(spec)

# This line is the magic fix for the "no known parent package" error:
florence_module.__package__ = "core.image" 

spec.loader.exec_module(florence_module)

print("it has run upto this point")

def run_test():
    # Now we access the class through the module we just loaded
    vision_model = florence_module.Florence_2Model(device="cpu")
    
    print("Loading model weights...")
    vision_model.load_model()

    # Path to your test image
    image_path = os.path.join(current_dir, "..", "Sample_Images", "image1.png")
    if not os.path.exists(image_path):
        print(f"Error: Place 'image1.png' in {os.path.dirname(image_path)}")
        return

    image = Image.open(image_path).convert("RGB")
    
    print("Generating caption...")
    result = vision_model.generate(image)
    
    print("-" * 30)
    print(f"RESULT: {result}")
    print("-" * 30)

    vision_model.unload_model()

if __name__ == "__main__":
    run_test()