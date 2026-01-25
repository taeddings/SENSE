import logging
from sense.config import ENABLE_VISION

class VisionInterface:
    def __init__(self):
        self.enabled = ENABLE_VISION
        self.model = None
        self.logger = logging.getLogger("VisionInterface")
        if self.enabled:
            self.logger.info("👁️ Vision System Enabled. Waiting for first usage to load model...")
        else:
            self.logger.info("🚫 Vision System Disabled (Mobile Mode).")

    def see(self, image_path):
        if not self.enabled:
            return "Vision is disabled on this device."
        
        # LAZY LOADING PATTERN
        # We only import the heavy logic INSIDE the function, not at module level.
        try:
            if self.model is None:
                self.logger.info("...Loading Vision Model (This may take RAM)...")
                # Import the harvested logic here
                from sense.vision import vision_process
                self.model = vision_process
                
            # Assuming the harvested 'vision_process' has a 'process' function or similar.
            # Based on the file name, it might be a module. We need to check what functions it exposes.
            # If inspection shows it's a script or class, we might need to adapt.
            # For now, we assume a generic entry point or placeholder.
            if hasattr(self.model, 'process_vision_info'):
                # This seems to be the main processing function
                return f"Vision module loaded. Ready to process using {self.model.process_vision_info.__name__}"
            elif hasattr(self.model, 'extract_vision_info'):
                return f"Vision module loaded. Ready to extract using {self.model.extract_vision_info.__name__}"
            else:
                 return "Vision logic loaded, but no known entry point found."

        except ImportError as e:
            self.logger.error(f"Failed to load Vision libs: {e}")
            self.enabled = False # Safety tripwire
            return f"Vision dependencies missing: {e}"
        except Exception as e:
            return f"Vision Error: {e}"
