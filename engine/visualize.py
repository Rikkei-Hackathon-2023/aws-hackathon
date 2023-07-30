import gradio as gr
import requests
from PIL import Image, ImageFile
from glob import glob
import functools
import time
import os

root = "/home/giang/aws_hackathon/assets/figures"
images = os.listdir(root)
images = [f for f in images if f.endswith(".png")]
ImageFile.LOAD_TRUNCATED_IMAGES = True


def refresh(file):
    while True:
        try:
            image = Image.open(os.path.join(root, file))
            return image
        except Exception:
            time.sleep(1)
            continue


# refresh1 = functools.partial(refresh, images[0])
# refresh2 = functools.partial(refresh, images[1])
# refresh3 = functools.partial(refresh, images[2])
# refresh4 = functools.partial(refresh, images[3])
# refresh5 = functools.partial(refresh, images[4])
refresh1 = functools.partial(refresh, os.path.join(root, "candles.png"))

with gr.Blocks(show_footer=False) as blocks:
    with gr.Row():
        with gr.Column():
            image1 = gr.Image(show_label=False)
            # image2 = gr.Image(show_label=False)
            # image5 = gr.Image(show_label=False)
        # with gr.Column():
            # image3 = gr.Image(show_label=False)
            # image4 = gr.Image(show_label=False)

    blocks.load(fn=refresh1, inputs=None, outputs=image1,
                show_progress=False, every=1)
    # blocks.load(fn=refresh2, inputs=None, outputs=image2,
    #             show_progress=False, every=1)
    # blocks.load(fn=refresh3, inputs=None, outputs=image3,
    #             show_progress=False, every=1)
    # blocks.load(fn=refresh4, inputs=None, outputs=image4,
    #             show_progress=False, every=1)
    # blocks.load(fn=refresh5, inputs=None, outputs=image5,
    #             show_progress=False, every=1)

blocks.queue(api_open=False)
blocks.launch()
