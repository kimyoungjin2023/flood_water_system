"""
모델 로딩 모듈 (v4.0)
- ResNet-18        : 침수 상황 분류 (파일명 변경)
- EfficientNet-B0  : 아동/성인, 성별, 차량 종류 (신규 + 파일명 변경)
- Keras Attention U-Net : 침수 Seg
- YOLO x5          : 탐지/Pose/차량부품/표지판/시선유도봉
"""
import os, warnings, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").setLevel(logging.WARNING)

import torch
import torch.nn as nn
import torchvision.models as tv
from ultralytics import YOLO
from pathlib import Path

from utils.config import MODEL_DIR, MODEL_FILES, CAR_CLS_NAMES


# ══════════════════════════════════════════════════════════════
#  PyTorch 모델 정의
# ══════════════════════════════════════════════════════════════

class ResNet18Classifier(nn.Module):
    """침수 상황 분류기 (ResNet-18, 2cls: normal/flood)"""
    def __init__(self):
        super().__init__()
        base    = tv.resnet18(weights=None)
        base.fc = nn.Linear(512, 2)
        self.model = base

    def forward(self, x):
        return self.model(x)


class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 기반 분류기
    child/gender: 2클래스
    car_cls: 5클래스 (bus/car-mini/car-pickup/car-sedan/car-suv)
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = tv.efficientnet_b0(weights=None)
        # classifier: [Dropout, Linear(1280→num_classes)]
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def load_checkpoint(self, path: str) -> list:
        """
        shape 일치 파라미터만 복사 (strict=False 안전 버전)
        EfficientNet-B0는 구조가 표준이므로 대부분 일치 예상
        """
        sd_ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        model_sd = self.state_dict()
        new_sd   = {}

        for k, v in sd_ckpt.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                new_sd[k] = v

        missing = [k for k in model_sd if k not in new_sd]
        model_sd.update(new_sd)
        self.load_state_dict(model_sd, strict=True)
        return missing


# ══════════════════════════════════════════════════════════════
#  Keras Flood Segmentation (Attention U-Net)
# ══════════════════════════════════════════════════════════════

def _load_keras_seg():
    try:
        import tensorflow as tf
        import keras

        @keras.saving.register_keras_serializable()
        class EncoderBlock(keras.layers.Layer):
            def __init__(self, filters, rate=0.1, pooling=True, **kw):
                super().__init__(**kw)
                self.filters = filters; self.rate = rate; self.pooling = pooling
                self.c1   = keras.layers.Conv2D(filters, 3, padding="same",
                                activation="relu", kernel_initializer="he_normal", name="c1")
                self.drop = keras.layers.Dropout(rate, name="drop")
                self.c2   = keras.layers.Conv2D(filters, 3, padding="same",
                                activation="relu", kernel_initializer="he_normal", name="c2")
                self.pool = keras.layers.MaxPooling2D(name="pool") if pooling else None
            def call(self, x, training=False):
                x = self.c1(x); x = self.drop(x, training=training); x = self.c2(x)
                return (self.pool(x), x) if self.pooling else x
            def get_config(self):
                cfg = super().get_config()
                cfg.update({"filters":self.filters,"rate":self.rate,"pooling":self.pooling})
                return cfg

        @keras.saving.register_keras_serializable()
        class AttentionGate(keras.layers.Layer):
            """
            h5 분석 기반 정확한 구조:
              normal(3×3,stride=1) on g(작은) → (h,w,filters)
              down  (3×3,stride=2) on x(큰)   → (h,w,filters)  downsample
              BN → relu → learn(1×1,sigmoid) → psi(h,w,1)
              resample(2×UP) → psi_up(2h,2w,1)
              output: x * psi_up
            """
            def __init__(self, filters, bn=True, **kw):
                super().__init__(**kw)
                self.filters = filters; self.bn_flag = bn
                self.normal   = keras.layers.Conv2D(filters, 3, padding="same",    name="normal")
                self.down     = keras.layers.Conv2D(filters, 3, strides=2, padding="same", name="down")
                self.BN       = keras.layers.BatchNormalization(name="BN") if bn else None
                self.learn    = keras.layers.Conv2D(1, 1, padding="same",
                                                    activation="sigmoid", name="learn")
                self.resample = keras.layers.UpSampling2D(size=(2,2), name="resample")
                self.relu_act = keras.layers.Activation("relu")
            def call(self, inputs, training=False):
                g, x  = inputs[0], inputs[1]
                gf    = self.normal(g)
                xf    = self.down(x)
                psi   = self.relu_act(gf + xf)
                if self.BN: psi = self.BN(psi, training=training)
                psi   = self.learn(psi)
                return x * self.resample(psi)
            def get_config(self):
                cfg = super().get_config()
                cfg.update({"filters":self.filters,"bn":self.bn_flag})
                return cfg

        @keras.saving.register_keras_serializable()
        class DecoderBlock(keras.layers.Layer):
            def __init__(self, filters, rate=0.1, **kw):
                super().__init__(**kw)
                self.filters = filters; self.rate = rate
                self.up  = keras.layers.UpSampling2D(size=(2,2), name="up")
                self.cat = keras.layers.Concatenate()
                class Net(keras.layers.Layer):
                    def __init__(self, f, r, **kw2):
                        super().__init__(**kw2)
                        self.c1   = keras.layers.Conv2D(f,3,padding="same",activation="relu",
                                        kernel_initializer="he_normal",name="c1")
                        self.drop = keras.layers.Dropout(r)
                        self.c2   = keras.layers.Conv2D(f,3,padding="same",activation="relu",
                                        kernel_initializer="he_normal",name="c2")
                    def call(self, x, training=False):
                        return self.c2(self.drop(self.c1(x), training=training))
                    def get_config(self): return {}
                self.net = Net(filters, rate, name="net")
            def call(self, inputs, training=False):
                g, skip = inputs[0], inputs[1]
                return self.net(self.cat([self.up(g), skip]), training=training)
            def get_config(self):
                cfg = super().get_config()
                cfg.update({"filters":self.filters,"rate":self.rate})
                return cfg

        def dice_coef(y_true, y_pred, smooth=1e-6):
            yt = tf.reshape(y_true, [-1]); yp = tf.reshape(y_pred, [-1])
            i  = tf.reduce_sum(yt * yp)
            return (2.*i+smooth)/(tf.reduce_sum(yt)+tf.reduce_sum(yp)+smooth)

        return keras.models.load_model(
            f"{MODEL_DIR}/{MODEL_FILES['flood_seg']}",
            custom_objects={"EncoderBlock":EncoderBlock,"AttentionGate":AttentionGate,
                            "DecoderBlock":DecoderBlock,"dice_coef":dice_coef},
            compile=False,
        )
    except Exception as e:
        print(f"       ⚠️  Keras Seg 로드 실패: {e}")
        return None


# ══════════════════════════════════════════════════════════════
#  통합 모델 로더
# ══════════════════════════════════════════════════════════════

class ModelLoader:
    def __init__(self):
        self.M: dict = {}
        self._load_all()

    def _load_all(self):
        sep = "─" * 60
        print(f"\n╔{sep}╗")
        print(f"║{'  모델 로딩  v4.0':^60}║")
        print(f"╠{sep}╣")
        self._load_situation()
        self._load_flood_seg()
        self._load_yolo()
        self._load_person_classifiers()
        self._load_car_classifier()
        print(f"╚{sep}╝\n")

    def _try_load(self, key: str, model: nn.Module,
                  fname: str, desc: str):
        path = f"{MODEL_DIR}/{fname}"
        if not Path(path).exists():
            print(f"║    ⚠️  파일없음: {fname}")
            self.M[key] = None
            return
        try:
            missing = model.load_checkpoint(path)
            model.eval()
            self.M[key] = model
            print(f"║    ✅ {fname}  ({desc})")
            if missing:
                print(f"║       └ shape불일치 skip: {len(missing)}개")
        except Exception as e:
            self.M[key] = None
            print(f"║    ❌ {fname}: {e}")

    def _load_situation(self):
        print("║  [1/5] 침수 상황 분류기 (ResNet-18) ...")
        fname = MODEL_FILES["situation"]
        path  = f"{MODEL_DIR}/{fname}"
        if not Path(path).exists():
            print(f"║    ⚠️  파일없음: {fname}")
            self.M["situation"] = None
            return
        try:
            m  = ResNet18Classifier()
            sd = torch.load(path, map_location="cpu", weights_only=False)
            # state_dict에 'model.' prefix 추가
            new_sd = {f"model.{k}": v for k, v in sd.items()}
            m.load_state_dict(new_sd)
            m.eval()
            self.M["situation"] = m
            print(f"║    ✅ {fname}  (ResNet-18, 2cls)")
        except Exception as e:
            self.M["situation"] = None
            print(f"║    ❌ {fname}: {e}")

    def _load_flood_seg(self):
        print("║  [2/5] Flood Semantic Segmentation ...")
        model = _load_keras_seg()
        self.M["flood_seg"] = model
        if model:
            print("║    ✅ data_add_flood_sementic_segmentation.keras  (Attention U-Net 256×256)")
        else:
            print("║    ⚠️  Seg 없음")

    def _load_yolo(self):
        print("║  [3/5] YOLO 계열 ...")
        cfgs = [
            ("yolo_main", MODEL_FILES["yolo_main"], "객체탐지(사람/차량)"),
            ("yolo_pose", MODEL_FILES["yolo_pose"], "Pose 17kp"),
            ("yolo_car",  MODEL_FILES["yolo_car"],  "차량부품 Seg"),
            ("yolo_sign", MODEL_FILES["yolo_sign"], "표지판"),
            ("yolo_tube", MODEL_FILES["yolo_tube"], "시선유도봉"),
        ]
        for key, fname, desc in cfgs:
            path = f"{MODEL_DIR}/{fname}"
            try:
                self.M[key] = YOLO(path)
                print(f"║    ✅ {fname}  ({desc})")
            except Exception as e:
                self.M[key] = None
                print(f"║    ❌ {fname}: {e}")

    def _load_person_classifiers(self):
        print("║  [4/5] 사람 속성 분류기 (EfficientNet-B0) ...")
        for key, cls_n, desc in [
            ("child",  2, "아동/성인"),
            ("gender", 2, "성별"),
        ]:
            fname = MODEL_FILES[key]
            self._try_load(key, EfficientNetB0Classifier(cls_n), fname, desc)

    def _load_car_classifier(self):
        print("║  [5/5] 차량 종류 분류기 (EfficientNet-B0, 5cls) ...")
        fname = MODEL_FILES["car_cls"]
        self._try_load(
            "car_cls",
            EfficientNetB0Classifier(num_classes=len(CAR_CLS_NAMES)),
            fname,
            f"5cls: {', '.join(CAR_CLS_NAMES)}",
        )
