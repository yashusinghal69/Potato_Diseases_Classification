import gradio as gr
import tensorflow as tf
#import numpy as np

model=tf.keras.models.load_model("final_model_99%.keras")
class_names =['Potato___Early_blight','Potato___Late_blight','Potato___healthy']

def predict(img):
    img_array = img.reshape(-1,128,128,3)
    # img_array = tf.keras.preprocessing.image.img_to_array(img)
    #img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)[0]

#     predicted_class = class_names[np.argmax(predictions[0])]
#     confidence = round(100 * (np.max(predictions[0])), 2)
    return {class_names[i]: float(predictions[i]) for i in range(3)}


article = "<h3>How to Use:</h3> " \
          "<ul><li>Click on the Upload button to upload an image,you can also drag the image to the upload box.</li> " \
          "<li>Choose a Image from your computer</li>" \
          "<li>Click on the 'Submit' button. <strong>Voila!</strong>. " \
          "and labels will be displayed on screen.</li></ul>"


# with gr.Blocks() as demo:
demo = gr.Interface(fn=predict,
                    inputs=[gr.Image(label="Upload an image",show_share_button=True,interactive=True,show_download_button=True)],
                    outputs=[gr.Label(num_top_classes=3,label="Predictions")],
                    title="Potato Disease Classification",
                    description="",
                    examples=['sample_images\potato_early_blight.JPG','sample_images\potato_healty.JPG','sample_images\potato_late_blight.JPG'],
                    allow_flagging="never",
                    article=article,
                    theme=gr.themes.Soft(),
                    )              

demo.launch(debug=True,share=True)
     