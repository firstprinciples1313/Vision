# Florence-2 Model Implementation - Common Issues & Solutions

## Issue 1: Abstract Class Instantiation Error
**Error Message:**
```
TypeError: Can't instantiate abstract class Florence_2Model without an implementation for abstract method 'generate'
```

**Root Cause:**
The method name was misspelled as `geneerate()` instead of `generate()`, so it didn't override the abstract method from `BaseImageModel`.

**Solution:**
```python
# WRONG:
def geneerate(self, image, task_prompt: str = "<DETAILED_CAPTION>"):
    ...

# CORRECT:
def generate(self, image, task_prompt: str = "<DETAILED_CAPTION>"):
    ...
```

---

## Issue 2: Method Signature Mismatch
**Error Message:**
Missing method implementation due to signature mismatch

**Root Cause:**
The `load_model()` method in `Florence_2Model` had an extra parameter `model_id` that wasn't in the base class `BaseImageModel.load_model()`, causing Python not to recognize it as an override.

**Solution:**
```python
# WRONG:
def load_model(self, model_id):
    self.processor = AutoProcessor.from_pretrained(self.model_id, ...)

# CORRECT:
def load_model(self):
    self.processor = AutoProcessor.from_pretrained(self.model_id, ...)
```

---

## Issue 3: Typo in Attribute Name
**Error Message:**
Silent bug - attribute never set correctly

**Root Cause:**
In `unload_model()`, the code had `self.mode = None` instead of `self.model = None`.

**Solution:**
```python
# WRONG:
del self.model
del self.processor
self.mode = None  # typo!

# CORRECT:
del self.model
del self.processor
self.model = None
```

---

## Issue 4: Network Access Blocked by Firewall
**Error Message:**
```
[Errno 11002] getaddrinfo failed
RuntimeError: Cannot send a request, as the client has been closed.
OSError: We couldn't connect to 'https://huggingface.co' to load the files
```

**Root Cause:**
University firewall (Fortinet) blocks access to huggingface.co domain.

**Solution:**
- Switch to a network without firewall restrictions (personal WiFi, hotspot)
- Use university VPN if available
- Download model on an unrestricted network and transfer cache files

---

## Issue 5: local_files_only Flag Misuse
**Error Message:**
```
huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled.
```

**Root Cause:**
Set `local_files_only=True` to bypass network issues, but model files weren't cached yet, so it couldn't find them locally.

**Solution:**
```python
# For FIRST RUN (need to download):
processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
# local_files_only is False by default - allows download

# For SUBSEQUENT RUNS (model cached, want offline):
processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, local_files_only=True)
# Only uses cached files, no network needed
```

---

## Issue 6: Transformers Version Compatibility
**Error Message:**
```
AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'
```

**Root Cause:**
`transformers==5.5.0` is not compatible with Florence-2 model architecture. Newer versions have the required attributes.

**Solution:**
```bash
pip install --upgrade "transformers>=4.45.0"
```

Update requirements.txt:
```
transformers>=4.45.0  # was: transformers==5.5.0
```

---

## Issue 7: Florence-2 Custom Configuration Bug (Well-Known Issue)
**Error Message:**
```
AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'
```
*Persists even after updating transformers*

**Root Cause:**
The Florence-2 model's custom configuration file (`configuration_florence2.py`) references an attribute `forced_bos_token_id` that doesn't exist as a class attribute by default. This is a known bug in the Florence-2 model repository. The custom config file is downloaded from Hugging Face and tries to access this attribute through `__getattribute__`, which fails.

**Known Issue Status:** ✅ **WELL-KNOWN BUG** - This affects many users trying to use Florence-2 with certain transformers versions.

**Solution:**
Implement error handling with fallback to separate image processor and tokenizer:

```python
try:
    self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
except AttributeError as e:
    if "forced_bos_token_id" in str(e):
        print("Detected Florence-2 configuration bug, using workaround...")
        from transformers import AutoImageProcessor, AutoTokenizer
        self.processor = None
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
    else:
        raise
```

Then handle both cases in the `generate()` method.

---

## Complete Fix Checklist

- [ ] Fix method name typo: `geneerate` → `generate`
- [ ] Fix method signature: `load_model(self, model_id)` → `load_model(self)`
- [ ] Fix attribute typo: `self.mode = None` → `self.model = None`
- [ ] Ensure internet access to Hugging Face (disable firewall or use VPN)
- [ ] Use `local_files_only=False` (default) for first run
- [ ] Upgrade transformers: `pip install --upgrade "transformers>=4.39.0"`
- [ ] Clear entire cache: `Remove-Item -Recurse -Force "C:\Users\{user}\.cache\huggingface\"`
- [ ] Implement error handling for Florence-2 configuration bug
- [ ] Add fallback to `AutoImageProcessor` + `AutoTokenizer` if `AutoProcessor` fails
- [ ] Update `generate()` method to handle both processor and separate components
- [ ] Run script once to download & cache model
- [ ] After caching, can use `local_files_only=True` for offline mode

---

## Hugging Face Cache Location
```
C:\Users\FirstPrinciples\.cache\huggingface\hub\
```

The model files are automatically cached here after first download. You can copy this entire folder to another machine to avoid re-downloading.
