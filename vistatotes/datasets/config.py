"""Dataset configurations for VectoryTones demo datasets."""

DEMO_DATASETS = {
    # ── Sounds ──────────────────────────────────────────────────────────────
    "nature_sounds": {
        "label": "Animal & Nature Sounds",
        "description": "Bird calls, frog croaks, insect buzzes, rain, wind, and other outdoor sounds from the ESC-50 collection.",
        "categories": [
            "chirping_birds",
            "crow",
            "frog",
            "insects",
            "rain",
            "sea_waves",
            "thunderstorm",
            "wind",
            "water_drops",
            "crickets",
        ],
        "media_type": "audio",
    },
    "city_sounds": {
        "label": "City & Indoor Sounds",
        "description": "Traffic, machinery, appliances, and the daily sounds of human environments from the ESC-50 collection.",
        "categories": [
            "car_horn",
            "siren",
            "engine",
            "train",
            "helicopter",
            "vacuum_cleaner",
            "washing_machine",
            "clock_alarm",
            "keyboard_typing",
            "door_wood_knock",
        ],
        "media_type": "audio",
    },
    # ── Images ──────────────────────────────────────────────────────────────
    "animals_images": {
        "label": "Animals & Wildlife",
        "description": "400 photographs of birds, cats, dogs, horses, deer, and frogs sourced from the CIFAR-10 dataset.",
        "categories": [
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
        ],
        "media_type": "image",
        "source": "cifar10_sample",
    },
    "vehicles_images": {
        "label": "Vehicles & Transport",
        "description": "400 photographs of airplanes, cars, ships, and trucks sourced from the CIFAR-10 dataset.",
        "categories": [
            "airplane",
            "automobile",
            "ship",
            "truck",
        ],
        "media_type": "image",
        "source": "cifar10_sample",
    },
    # ── Videos ──────────────────────────────────────────────────────────────
    "activities_video": {
        "label": "Personal Activities",
        "description": "Short clips of everyday personal activities like grooming, playing instruments, and yo-yo from UCF-101.",
        "categories": [
            "ApplyEyeMakeup",
            "ApplyLipstick",
            "BrushingTeeth",
            "Drumming",
            "YoYo",
        ],
        "media_type": "video",
        "source": "ucf101",
    },
    "sports_video": {
        "label": "Sports & Exercise",
        "description": "Short clips of physical activities including cliff diving, jump rope, push-ups, and tai chi from UCF-101.",
        "categories": [
            "CliffDiving",
            "HandstandWalking",
            "JumpRope",
            "PushUps",
            "TaiChi",
        ],
        "media_type": "video",
        "source": "ucf101",
    },
    # ── Texts ───────────────────────────────────────────────────────────────
    "world_news": {
        "label": "World & Business News",
        "description": "Paragraphs drawn from international news and business articles in the 20 Newsgroups collection.",
        "categories": [
            "world",
            "business",
        ],
        "media_type": "paragraph",
        "source": "ag_news_sample",
    },
    "sports_science_news": {
        "label": "Sports & Science News",
        "description": "Paragraphs drawn from sports coverage and science journalism in the 20 Newsgroups collection.",
        "categories": [
            "sports",
            "science",
        ],
        "media_type": "paragraph",
        "source": "ag_news_sample",
    },
}
