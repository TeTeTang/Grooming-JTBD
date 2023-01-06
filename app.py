import streamlit as st
import pandas as pd

import bert_view
import gpt_ft


def load_file():
    uploaded_file = st.file_uploader(label="Please upload your excel file and specify the column name as 'text'")
    if uploaded_file is not None:
        # input_data = uploaded_file.getvalue()
        # st.text(input_data)
        df= pd.read_excel(uploaded_file)
        return df
    else:
        st.write("Can't get your file")
    

def main():
    st.title('Grooming JTBD demo')
    data = load_file()
    predictions, _ = bert_view.inference(data)
    text_list = data["text"].tolist()
    ## Optional: Write the input text and bert predictions into a dataframe
    # st.write(pd.DataFrame({
    # 'input_text': text_list,
    # 'BERT_prediction': predictions,
    # }))
    qualified_texts = []
    jtbd_pred_rslt = []
    for i, val in enumerate(predictions):
        if val == 1:
            qualified_text = text_list[i]
            qualified_texts.append(qualified_text)
    for qt in qualified_texts:
        jtbd_pred = gpt_ft.gpt_pred(qt)
        jtbd_pred_rslt.append(jtbd_pred)
        st.write(pd.DataFrame({
            'related_text': qualified_texts,
            'JTBD_prediction': jtbd_pred_rslt,
        }))


if __name__ == '__main__':
    main()