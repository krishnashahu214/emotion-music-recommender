from transformers import pipeline

# classifier = pipeline(
#     "text-classification",
#     model="j-hartmann/emotion-english-distilroberta-base"
# )

# def get_text_emotion(text):
#     result = classifier(text)
#     return result[0]['label']


classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

def get_text_emotion(text):
    result = classifier(text)
    label = result[0]['label']
    confidence = result[0]['score']
    
    return label, confidence