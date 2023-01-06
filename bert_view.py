import torch 
from torch.cuda import is_available

from simpletransformers.classification import ClassificationModel, ClassificationArgs

model_class_map = {"ClassificationModel": "Classification Model"}

# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=5)
# Load the trained model
loaded_model = ClassificationModel("bert", "bert_model", args=model_args)  

# Inference 
def inference(df):
    input_text = df["text"].tolist()
    predictions, logprobs = loaded_model.predict(input_text)
    return predictions, logprobs



# text_relevant = []
# for idx, text in enumerate(input_list):
#     t = (text, predictions[idx])
#     text_relevant.append(t)

