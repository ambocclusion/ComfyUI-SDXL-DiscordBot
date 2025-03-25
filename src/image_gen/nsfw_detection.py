from nudenet import NudeDetector

from src.util import read_config

class NsfwDetector:
    def __init__(self):
        config = read_config()

        if config["NSFW_DETECTION"]["NUDE_DETECTOR_MODEL_PATH"] != "None":
            self.detector = NudeDetector(
                model_path=config["NSFW_DETECTION"]["NUDE_DETECTOR_MODEL_PATH"],
                inference_resolution=int(config["NSFW_DETECTION"]["NUDE_DETECTOR_INFERENCE_RESOLUTION"])
            )
        else:
            self.detector = None
            
        self.class_blacklist = config["NSFW_DETECTION"]["NUDE_DETECTOR_CLASS_BLACKLIST"].split(",")
        self.term_blacklist = config["NSFW_DETECTION"]["NSFW_TERM_BLACKLIST"].split(",")
        
        
    def detect_from_image(self, path):
        if self.detector == None:
            return False

        results = self.detector.detect(path)

        for result in results:
            if result["class"] in self.class_blacklist:
                print("NudeDetector detected nudity of type: " + result["class"])
                return True
        
        return False

    def detect_from_prompt(self, prompt):
        prompt_words = ''.join(c for c in prompt if c.isalpha() or c.isspace()).split( )

        for word in prompt_words:
            if word in self.term_blacklist:
                print("Blacklisted word found in prompt: " + word)
                return True

        return False

def check_nsfw(path, prompt):
    nsfw_detector = NsfwDetector()
    try:
        is_nsfw = nsfw_detector.detect_from_image(path) or nsfw_detector.detect_from_prompt(prompt)
    except:
        print("Nudity detection not supported for this format")
        is_nsfw = nsfw_detector.detect_from_prompt(prompt)

    return is_nsfw