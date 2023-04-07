import streamlit as st
from img_classification import teachable_machine_classification
from PIL import Image, ImageOps
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# authentification
with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')
if authentication_status:
    authenticator.logout('Logout', 'main')
    page = st.sidebar.selectbox("探索或预测", ("肺炎x_ray图像分类","ff"))

    if page == "肺炎x_ray图像分类":
        st.title("使用谷歌的可教机器进行图像分类")
        st.write("Google Teachable machine"" [link](https://teachablemachine.withgoogle.com/train/image)")
        st.header("肺炎x_ray")
        st.text("上传肺x_ray图片")

        uploaded_file = st.file_uploader("选择..", type=["jpg","png","jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='上传了图片。', use_column_width=True)
            st.write("")
            st.write("分类...")
            label = teachable_machine_classification(image, 'pneumonia__x_ray_image_classify_normal_vs_penumonia.h5')
            if label == 0:
                st.write("正常")
            else:
                st.write("肺炎")

        st.text("类:正常,肺炎")

        # 0 normal
        # 1 pneumonia

elif authentication_status == False:
    st.error("用户名/密码不正确")
elif authentication_status == None:
    st.warning('请输入您的用户名和密码')








