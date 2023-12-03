import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pandas as pd

# Streamlit UI layout settings
st.set_page_config(page_title="Transformers UI", page_icon="🤖", layout="wide")

# Sidebar for model selection and settings
with st.sidebar:
    st.title("Transformers UI")

    # Input for model identifier or path
    model_identifier_or_path = st.text_input("预训练模型路径:")

    # Input for tokenizer path
    tokenizer_identifier_or_path = st.text_input("分词器标识符或路径:")

    # Input for dataset path
    dataset_path = st.text_input("数据集文件路径:")

    debug_mode = st.checkbox("启用调试模式")

# Main content area
with st.container():
    st.header("选择的模型")

    # Load selected model and tokenizer
    if model_identifier_or_path:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_identifier_or_path)

            # Load tokenizer if provided, or use default
            if tokenizer_identifier_or_path:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_identifier_or_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_identifier_or_path)

            text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
            st.success("成功加载指定路径或标识符的预训练模型!")

            # Load dataset if provided
            if dataset_path:
                try:
                    st.write("成功加载指定路径的数据集文件!")
                    df = pd.read_csv(dataset_path)  # Adjust based on the file type
                    st.write("数据集预览:")
                    st.write(df.head())
                except Exception as e:
                    st.error(f"加载数据集文件时出现错误: {str(e)}")

            # User input for generating text
            user_input = st.text_input("您的输入:")
            generate_button = st.button("生成")

            # Display generated text
            if generate_button:
                if debug_mode:
                    st.write(f"调试模式: {debug_mode}")
                generated_text = text_gen(user_input, max_length=100)
                st.success("生成的文本:")
                st.write(generated_text[0]["generated_text"])

        except Exception as e:
            st.error(f"加载模型文件时出现错误: {str(e)}")

    # Save chat history
    if st.button("保存聊天记录"):
        chat_history = st.text_area("聊天记录:", "")
        st.write("聊天记录已保存!")

    # Display 720p dialog
    st.write("720p 对话:")
    st.text_area("", height=300)

    # Background selection
    st.sidebar.subheader("背景设置")
    background_color = st.sidebar.color_picker("背景颜色", "#ffffff")
    st.markdown(
        f"""<style>
        .reportview-container {{
            background-color: {background_color};
        }}
        </style>""",
        unsafe_allow_html=True,
    )
