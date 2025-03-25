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
            
        self.trigger_classes = config["NSFW_DETECTION"]["NUDE_DETECTOR_TRIGGER_CLASSES"].split(",")
        self.trigger_words = config["NSFW_DETECTION"]["NSFW_TRIGGER_WORDS"].split(",")
        
        
    def detect_from_image(self, path):
        if self.detector == None:
            return False

        results = self.detector.detect(path)

        print(results)

        for result in results:
            if result["class"] in self.trigger_classes:
                return True
        
        return False

    def detect_from_prompt(self, prompt):
        prompt_words = ''.join(c for c in prompt if c.isalpha() or c.isspace()).split( )

        for word in prompt_words:
            if word in self.trigger_words:
                print("naughty word: " + word)
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