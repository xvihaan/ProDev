import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 컬러 이미지를 로드하기
img = np.array(Image.open("/content/skyandbreeze.jpg"))

# RGB 채널로 각각 분리하여 변수에 저장하기
R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]

# 각 채널별로 히스토그램 계산하기
histo_R, bins = np.histogram(R.flatten(), 256, [0,255])
histo_G, bins = np.histogram(G.flatten(), 256, [0,255])
histo_B, bins = np.histogram(B.flatten(), 256, [0,255])

# 각 채널별로 누적(cdf) 히스토그램 계산하기
cdf_R = histo_R.cumsum()
cdf_G = histo_G.cumsum()
cdf_B = histo_B.cumsum()

# 히스토그램 평활화 수식 적용하기
cdf_R = 255 * cdf_R / cdf_R[-1]
cdf_G = 255 * cdf_G / cdf_G[-1]
cdf_B = 255 * cdf_B / cdf_B[-1]

# 각 채널별로 평활화된 이미지 생성하기
img_R = np.interp(R.flatten(), bins[:-1], cdf_R)
img_G = np.interp(G.flatten(), bins[:-1], cdf_G)
img_B = np.interp(B.flatten(), bins[:-1], cdf_B)

# 생성된 이미지를 RGB 채널로 다시 합쳐 최종 이미지를 생성하기
img_R = img_R.reshape(R.shape)
img_G = img_G.reshape(G.shape)
img_B = img_B.reshape(B.shape)
img_eq = np.stack((img_R, img_G, img_B), axis=2)
img_eq = img_eq.astype(np.uint8)

# subplot 만들기
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,5))

# 원본 이미지 출력하기
axes[0, 0].imshow(img)
axes[0, 0].set_title('Original Image')

# 원본 이미지 히스토그램 표시하기
axes[1, 0].hist(img.ravel(), 256, [0,255], color='gray')
axes[1, 0].set_title('Original Image Histogram')

# 평활화된 이미지 표시하기
axes[0, 1].imshow(img_eq)
axes[0, 1].set_title('Equalized Image')

# 평활화된 이미지 히스토그램 표시하기
axes[1, 1].hist(img_eq.ravel(), 256, [0,255], color='gray')
axes[1, 1].set_title('Equalized Image Histogram')

# 전체 subplot 조절하기
plt.tight_layout()

# 전체 plot 보여주기
plt.show()
