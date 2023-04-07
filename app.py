import streamlit as st
from img_classification import teachable_machine_classification
from PIL import Image, ImageOps
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

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
    page = st.sidebar.selectbox("探索或预测", ("将图像放大为高清","肺炎x_ray图像分类"))

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
    elif page =="将图像放大为高清":
        st.title("使用 ESGAN 放大图像")
        st.write("ESGAN 安装"" [link](https://github.com/xinntao/ESRGAN)")
        st.write("ESGAN 模型下载"" [link](https://drive.google.com/drive/u/0/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY)")
        st.header("将图像放大为高清")
        st.text("上传图片")

        model_path = './RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
        # device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
        device = torch.device('cpu')

        # test_img_folder = 'LR/*'
        uploaded_file = st.file_uploader("选择..", type=["jpg","png","jpeg"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption='上传了图片。', use_column_width=True)
            st.write("")
            st.write("")
            st.write("放大图像，大约等待时间：1 分钟,请稍候...")

            rrdb_esrgan_model = arch.RRDBNet(3, 3, 64, 23, gc=32)
            rrdb_esrgan_model.load_state_dict(torch.load(model_path), strict=True)
            rrdb_esrgan_model.eval()
            rrdb_esrgan_model = rrdb_esrgan_model.to(device)

            idx = 0

            # img = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3) * 1.0 / 255
            # uploaded_file = st.file_uploader("Upload Image")
            # image = Image.open(uploaded_file)
            # st.image(image, caption='Input', use_column_width=True)
            img = np.array(img)* 1.0 / 255
            # cv2.imwrite('out.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_LR = img.unsqueeze(0)
            img_LR = img_LR.to(device)

            with torch.no_grad():
                output = rrdb_esrgan_model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = torch.tensor((output * 255.0).round())
            fig1 = plt.figure(figsize=(14,8))

            fig1.suptitle("Upscaled image")
            plt.imshow(np.transpose(vutils.make_grid(output, padding=2, normalize=True), (0,1, 2)))  

            st.pyplot(fig1)


elif authentication_status == False:
    st.error("用户名/密码不正确")
elif authentication_status == None:
    st.warning('请输入您的用户名和密码')








