import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import psutil
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
import os

# Define the generate_window_image function
def generate_window_image(width, height, window_color):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    window_size = min(width, height) // 2
    window_x = (width - window_size) // 2
    window_y = (height - window_size) // 2
    draw.rectangle([window_x, window_y, window_x + window_size, window_y + window_size], fill=window_color)
    return image

# Function to select pretrained model and show current directory
def select_pretrained_model():
    st.subheader("选择预训练模型")

    # Show current directory
    current_directory = os.getcwd()
    st.write(f"当前文件夹: {current_directory}")

    # Get all files and directories in the current directory
    files_and_dirs_in_directory = os.listdir(current_directory)

    # Separate files and directories
    files_in_directory = [f for f in files_and_dirs_in_directory if os.path.isfile(os.path.join(current_directory, f))]

    # Input for model path
    model_identifier_or_path = st.text_input("手动输入预训练模型路径:")
    if not model_identifier_or_path:
        # Input for selecting pretrained model
        selected_model = st.selectbox("选择预训练模型或文件夹:", files_and_dirs_in_directory)

        # If the selected item is a file, return its path
        if os.path.isfile(os.path.join(current_directory, selected_model)):
            model_identifier_or_path = selected_model
        else:
            # If the selected item is a directory, display its contents
            model_identifier_or_path = st.selectbox("选择文件:", [f for f in os.listdir(os.path.join(current_directory, selected_model))])

    return model_identifier_or_path

# Streamlit UI layout settings
st.set_page_config(page_title="Transformers UI", page_icon="🤖", layout="wide")

# Sidebar for model selection and settings
with st.sidebar:
    st.title("Transformers UI")

    # Input for model identifier or path
    model_identifier_or_path = select_pretrained_model()

    # Input for tokenizer path
    tokenizer_identifier_or_path = st.text_input("分词器标识符或路径:")

    # Input for dataset path
    dataset_path = st.text_input("数据集文件路径:")

    debug_mode = st.checkbox("启用调试模式")

    # Upload virtual assistant image
    virtual_assistant_image = st.sidebar.file_uploader("上传虚拟助手图像文件（支持JPG、PNG等格式）", type=["jpg", "png"])

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
                    table = pq.read_table(dataset_path)
                    df = table.to_pandas()
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
    background_color = st.sidebar.color_picker("背景颜色", "#ffffff", key="background_color")
    st.markdown(
        f"""<style>
        .reportview-container {{
            background-color: {background_color};
        }}
        </style>""",
        unsafe_allow_html=True,
    )

    # CPU and Memory Monitor
    st.sidebar.subheader("性能监控")
    cpu_percent = psutil.cpu_percent(interval=1) / 100.0  # 将百分比转换为 0.0 到 1.0 的范围
    memory_info = psutil.virtual_memory()

    st.sidebar.text(f"CPU 使用率: {cpu_percent * 100}%")
    st.sidebar.progress(cpu_percent)

    st.sidebar.text(f"内存使用率: {memory_info.percent}%")
    st.sidebar.progress(memory_info.percent / 100.0)  # 将百分比转换为 0.0 到 1.0 的范围

# Main content area
with st.container():
    # User input for window color
    window_color = st.color_picker("窗户颜色", "#87CEEB", key="window_color")  # 默认为天蓝色

    # Display generated window image
    window_image = generate_window_image(400, 400, window_color)
    st.image(window_image, caption="生成的窗户图片", use_column_width=True)

    # User input for image generation
    st.subheader("生成图片输入窗口")
    user_image_input = st.file_uploader("上传图片文件（支持JPG、PNG等格式）", type=["jpg", "png"])

    # Check if user has uploaded an image
    if user_image_input is not None:
        # Process the user-uploaded image
        user_image = Image.open(user_image_input)
        st.image(user_image, caption="上传的图片", use_column_width=True)

        # You can add your image processing logic here if needed

        # Example: Display a message
        st.write("图片生成成功！")

# Header for virtual assistant
st.header("虚拟人助手")

# Display virtual assistant image
if virtual_assistant_image is not None:
    st.image(virtual_assistant_image, caption="虚拟人助手", use_column_width=True)

# User input for interacting with the virtual assistant
user_input_va = st.text_input("您好，有什么可以帮助您的吗？")

# Example: Virtual
