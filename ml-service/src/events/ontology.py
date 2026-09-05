# Target Ontology: 18 Non-Speech Audio Event Classes
TARGET_AUDIO_CLASSES = [
    "aircraft",
    "helicopter",
    "car",
    "bus",
    "train",
    "vehicle",
    "car_horn",
    "siren",
    "alarm",
    "dog",
    "crowd",
    "footsteps",
    "engine",
    "machinery",
    "speech",
    "music",
    "rain",
    "thunder"
]

CLASS_TO_ID = {cls: idx for idx, cls in enumerate(TARGET_AUDIO_CLASSES)}
ID_TO_CLASS = {idx: cls for idx, cls in enumerate(TARGET_AUDIO_CLASSES)}

def get_ontology_info() -> dict:
    return {
        "num_classes": len(TARGET_AUDIO_CLASSES),
        "classes": TARGET_AUDIO_CLASSES,
        "class_to_id": CLASS_TO_ID,
        "id_to_class": ID_TO_CLASS
    }
