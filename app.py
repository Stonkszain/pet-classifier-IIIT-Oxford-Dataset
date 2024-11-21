from fastai.vision.all import *
import gradio as gr
import timm

learn = load_learner('model.pkl')

categories = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

gr.Interface(fn=predict, inputs=gr.Image(height=512), outputs=gr.Label(num_top_classes=3)).launch()