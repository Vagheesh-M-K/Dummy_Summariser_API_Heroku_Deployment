## suppress warnings
from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore")

from transformers import pipeline
pipe = pipeline(task = "summarization", model = "Falconsai/text_summarization")
def summarize(input_text) :
    op = pipe(input_text)
    return op[0]['summary_text']

import gradio as gr
iface = gr.Interface(fn = summarize, inputs = "text", outputs = "text")
iface.launch(share = True)

##
# select a model from HF and bring the model here to jupyter notebook or VS Code
# using pipeline function

# create a function= summ that takes a long text and summarises using the model I brought IN
# to the VS Code/.ipynb and returns its summary

#import gradio, create gradio.interface that inputs = "text", outputs = "text"
# and uses the function summ

#.launch(share = True) 

