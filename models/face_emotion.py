# from deepface import DeepFace

# def get_face_emotion(image_path):
#     try:
#         result = DeepFace.analyze(
#             img_path=image_path,
#             actions=['emotion'],
#             enforce_detection=False
#         )
#         return result[0]['dominant_emotion']
#     except:
#         return "neutral"


from efficientnet.RAF_predict import predict, load_model

model = load_model()

def get_face_emotion(image_path):
    try:
        face_emotion, confidence = predict(image_path, model)
        return face_emotion, confidence
    except:
        return "neutral", 0.0