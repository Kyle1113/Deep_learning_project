# Deep Learning Project

# wardrobe wizard(recommend clothes)

---

## 주제 선정배경

최근에는 패션 시장의 지속적인 확장을 반영해 무신사, 29cm, 에이버리 등 다양한 패션 플랫폼이 빠르게 성장하고 있다. 고객은 종종 어려움에 직면하고 구매 결정을 내리는 데 상당한 시간과 노력을 투자합니다. 이러한 요구를 해결하기 위해 많은 패션 플랫폼에서는 추천 시스템을 사용합니다.

그러나 기존 추천 시스템은 일반적으로 장바구니 분석(연관 규칙 마이닝)과 같은 방법을 사용하여 고객이 이미 구매했거나 선택한 제품과 유사한 제품을 제안하는 경우가 많습니다. 이 접근 방식은 유사한 항목을 추천하는 경향이 있으며, 이는 추천 시스템의 진정한 가치를 감소시킵니다.

이러한 한계를 극복하기 위해 우리는 고객이 선택한 의류와 어울리는 의류 품목을 제안하여 진정으로 의미 있는 추천을 제공함으로써 쇼핑 경험을 향상시키는 새로운 추천 시스템을 제안합니다.

---

## Flow Chart

![deep_learning_flow](https://github.com/Kyle1113/Deep_learning_project/assets/168116920/deec9edd-64ea-4817-99d1-ac172a1a46fe)

---

## Data(https://github.com/Kyle1113/Deep_learning_project/blob/2a20adcc457a3a3dea77a596316550835a2e8ed5/data/data_load.py)

## UNet Model Architecture (Sementic segmentation)

The UNet model is a popular architecture for image segmentation tasks. It consists of two main parts: the encoder (downsampling path) and the decoder (upsampling path), with a bottleneck layer in between. Below is a detailed explanation of the UNet architecture used in this project.

### Encoder (Downsampling Path)

The encoder path consists of several convolutional layers followed by max-pooling layers. Each step in the encoder halves the spatial dimensions of the input while doubling the number of feature channels, enabling the network to learn complex features.

1. **Input Layer:**
   - `inputs = layers.Input(input_size)`
   - Takes input images of size `(256, 256, 3)`.

2. **First Block:**
   - Two convolutional layers with 64 filters each, followed by ReLU activation and padding.
   - `c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)`
   - `c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)`
   - Max-pooling layer to reduce spatial dimensions by half.
   - `p1 = layers.MaxPooling2D((2, 2))(c1)`

3. **Second Block:**
   - Two convolutional layers with 128 filters each.
   - `c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)`
   - `c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)`
   - Max-pooling layer.
   - `p2 = layers.MaxPooling2D((2, 2))(c2)`

4. **Third Block:**
   - Two convolutional layers with 256 filters each.
   - `c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)`
   - `c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)`
   - Max-pooling layer.
   - `p3 = layers.MaxPooling2D((2, 2))(c3)`

5. **Fourth Block:**
   - Two convolutional layers with 512 filters each.
   - `c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)`
   - `c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)`
   - Max-pooling layer.
   - `p4 = layers.MaxPooling2D((2, 2))(c4)`

### Bottleneck

The bottleneck layer is the bridge between the encoder and decoder. It captures the most compressed representation of the input image.

1. **Bottleneck Block:**
   - Two convolutional layers with 1024 filters each.
   - `c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)`
   - `c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)`

### Decoder (Upsampling Path)

The decoder path consists of upsampling layers that increase the spatial dimensions of the feature maps. Each upsampling step is followed by concatenation with the corresponding feature maps from the encoder (skip connections), and then by two convolutional layers.

1. **First Up Block:**
   - Upsampling layer followed by concatenation with the fourth block of the encoder.
   - `u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)`
   - `u6 = layers.concatenate([u6, c4])`
   - Two convolutional layers with 512 filters each.
   - `c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)`
   - `c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)`

2. **Second Up Block:**
   - Upsampling layer followed by concatenation with the third block of the encoder.
   - `u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)`
   - `u7 = layers.concatenate([u7, c3])`
   - Two convolutional layers with 256 filters each.
   - `c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)`
   - `c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)`

3. **Third Up Block:**
   - Upsampling layer followed by concatenation with the second block of the encoder.
   - `u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)`
   - `u8 = layers.concatenate([u8, c2])`
   - Two convolutional layers with 128 filters each.
   - `c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)`
   - `c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)`

4. **Fourth Up Block:**
   - Upsampling layer followed by concatenation with the first block of the encoder.
   - `u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)`
   - `u9 = layers.concatenate([u9, c1])`
   - Two convolutional layers with 64 filters each.
   - `c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)`
   - `c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)`

### Output Layer

The final layer of the network is a convolutional layer that reduces the number of feature channels to the number of classes  using a softmax activation function, producing the final segmentation map.
- `outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)`

### Model Compilation

Finally, the model is compiled with the input and output layers specified.

```python
model = models.Model(inputs=[inputs], outputs=[outputs])
return model
```

### Summary

The UNet model is effective for image segmentation tasks due to its symmetric structure and skip connections, which allow it to capture both high-level and low-level features. This particular implementation uses the following configurations:

- **Input Size:** `(256, 256, 3)`
- **Number of Classes:** 3 (top, buttom, background)
- **Activation Functions:** ReLU for intermediate layers and softmax for the output layer.
- **Skip Connections:** Help preserve spatial information by combining features from the encoder with those of the decoder.

This architecture is well-suited for tasks requiring precise localization and identification of objects within an image, making it ideal for fashion segmentation applications.

### Feature Extraction

### Simillarity Distance
