# def fuse_emotions(face_emotion, text_emotion):

#     if face_emotion == text_emotion:
#         return face_emotion

#     # prioritize text
#     return text_emotion

def fuse_emotions(f_confi, t_confi, face_emo, text_emo):
    if f_confi > t_confi:
        return face_emo
    
    elif t_confi > f_confi:
        return text_emo
    
    else :
        return face_emo