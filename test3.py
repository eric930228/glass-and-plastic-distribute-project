import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.utils import Sequence
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

# ==================== 固定隨機種子 ====================
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 确保日志和模型保存目录存在
LOG_ROOT = r"C:\Users\user\Desktop\t2\logs"
os.makedirs(LOG_ROOT, exist_ok=True)

# ==================== 自定义 AugmentGenerator (Mixup+CutMix) ====================
class AugmentGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size=32, alpha=0.6,
                 shuffle=True, mode='hybrid', cutmix_prob=0.5,
                 seed=None, mix_ratio=0.7):
        self.x = np.array(x_set)
        self.y = np.array(y_set)
        self.batch_size = batch_size
        self.alpha = alpha
        self.mode = mode
        self.cutmix_prob = cutmix_prob
        self.mix_ratio = mix_ratio
        self.rng = np.random.default_rng(seed)
        self.indexes = np.arange(len(self.x))
        if shuffle:
            self.rng.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def on_epoch_end(self):
        self.rng.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        x1 = self._load(self.x[batch_idx])
        y1 = self.y[batch_idx]
        perm = self.rng.permutation(batch_idx)
        x2 = self._load(self.x[perm])
        y2 = self.y[perm]

        mixed_x, mixed_y = [], []
        for im1, im2, lab1, lab2 in zip(x1, x2, y1, y2):
            if self.rng.random() > self.mix_ratio:
                mixed_x.append(im1)
                mixed_y.append(lab1)
            else:
                lam = self.rng.beta(self.alpha, self.alpha)
                if self.mode=='mixup' or (self.mode=='hybrid' and self.rng.random() > self.cutmix_prob):
                    new_im = lam * im1 + (1-lam) * im2
                    new_lab = lam * lab1 + (1-lam) * lab2
                else:
                    h, w, _ = im1.shape
                    cut_w = int(w * np.sqrt(1-lam))
                    cut_h = int(h * np.sqrt(1-lam))
                    cx = self.rng.integers(w)
                    cy = self.rng.integers(h)
                    x1_, y1_ = max(cx-cut_w//2,0), max(cy-cut_h//2,0)
                    x2_, y2_ = min(cx+cut_w//2,w), min(cy+cut_h//2,h)
                    im1[y1_:y2_, x1_:x2_] = im2[y1_:y2_, x1_:x2_]
                    area = (x2_-x1_)*(y2_-y1_)
                    lam_adj = 1 - area/(w*h)
                    new_im = im1
                    new_lab = lam_adj*lab1 + (1-lam_adj)*lab2

                mixed_x.append(new_im)
                mixed_y.append(new_lab)

        return np.stack(mixed_x), np.stack(mixed_y)

    def _load(self, paths):
        imgs = []
        for p in paths:
            img = cv2.resize(cv2.imread(p), (260,260))
            img = preprocess_input(img.astype(np.float32))
            imgs.append(img)
        return np.array(imgs)

# ==================== 模型构建 ====================
def build_model(num_classes):
    base = EfficientNetB1(weights='imagenet', include_top=False, input_shape=(260,260,3))
    for layer in base.layers:
        layer.trainable = False
    x = base.output
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, kernel_regularizer=l2(1e-2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, kernel_regularizer=l2(1e-2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base.input, outputs=out)

# ==================== 分层 GroupKFold ====================
def stratified_group_k_fold(groups, labels, k, seed=None):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    unique = np.unique(groups)
    grp_lbl = [ labels[np.where(groups==g)[0][0]] for g in unique ]
    for train_i, val_i in skf.split(unique, grp_lbl):
        train_mask = np.isin(groups, unique[train_i])
        val_mask   = np.isin(groups, unique[val_i])
        yield np.where(train_mask)[0], np.where(val_mask)[0]

# ==================== 绘图函数 ====================
def plot_fold_loss(histories, fold):
    plt.figure()
    for h in histories:
        plt.plot(h.history['loss'])
        plt.plot(h.history['val_loss'])
    plt.title(f'Fold {fold} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    legend_labels = []
    for i in range(len(histories)):
        legend_labels.extend([f'train{i+1}', f'val{i+1}'])
    plt.legend(legend_labels)

    plt.savefig(f'fold_{fold}_loss_curve.png')
    plt.close()

# ==================== 繪製每個 fold 的 Acc / Loss 總覽圖 ====================
def plot_fold_summary(fold_metrics):
    folds = list(range(1, len(fold_metrics['train_acc']) + 1))
    
    plt.figure(figsize=(15, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(folds, fold_metrics['train_acc'], marker='o', label='Train')
    plt.plot(folds, fold_metrics['val_acc'], marker='o', label='Val')
    plt.plot(folds, fold_metrics['test_acc'], marker='o', label='Test')
    plt.title('Accuracy per Fold')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(folds, fold_metrics['train_loss'], marker='o', label='Train')
    plt.plot(folds, fold_metrics['val_loss'], marker='o', label='Val')
    plt.plot(folds, fold_metrics['test_loss'], marker='o', label='Test')
    plt.title('Loss per Fold')
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("fold_train_val_test_summary.png")
    plt.close()

# ==================== 主程式 ====================
if __name__=="__main__":
    
    all_train_acc, all_val_acc, all_test_acc = [], [], []
    all_train_loss, all_val_loss, all_test_loss = [], [], []

    # 读数据
    le = LabelEncoder()
    base_dir = r"C:\Users\user\Desktop\t1"
    test_dir = r"C:\Users\user\Desktop\t3"
    n_splits = 5
    batch_size = 32

    records = []
    for cls in os.listdir(base_dir):
        for fn in os.listdir(os.path.join(base_dir,cls)):
            if fn.lower().endswith(('.jpg','png','jpeg')):
                grp = '_'.join(fn.split('_')[:2])
                records.append({
                    'filename': os.path.join(base_dir,cls,fn),
                    'class': cls, 'group': grp
                })
    df = pd.DataFrame(records)
    df['class_encoded'] = le.fit_transform(df['class'])
    # 去掉 group size<5
    grp_cnt = df.groupby('group').size()
    valid = grp_cnt[grp_cnt>=5].index
    df = df[df['group'].isin(valid)].reset_index(drop=True)

    # 测试集
    test_records = []
    for cls in os.listdir(test_dir):
        for fn in os.listdir(os.path.join(test_dir,cls)):
            if fn.lower().endswith(('.jpg','png','jpeg')):
                test_records.append({
                    'filename': os.path.join(test_dir,cls,fn),
                    'class': cls
                })
    test_df = pd.DataFrame(test_records)

    # 生成 val generator
    val_df = None  # 以后按 fold 生成
    test_gen = ImageDataGenerator(preprocessing_function=preprocess_input) \
        .flow_from_dataframe(test_df, x_col='filename', y_col='class',
                             target_size=(260,260), batch_size=batch_size,
                             class_mode='categorical', shuffle=False, seed=seed)

    # 分组标签
    grp_df = df.groupby('group')['class'].agg(lambda x: x.mode()[0]).reset_index()
    groups = grp_df['group'].values
    labels = grp_df['class'].values

    all_train_acc, all_val_acc, all_test_acc = [], [], []
    all_train_loss, all_val_loss, all_test_loss = [], [], []

    # K-fold
    for fold, (train_idx, val_idx) in enumerate(
        stratified_group_k_fold(groups, labels, n_splits, seed), 1):

        # 划分 train/val
        train_grps = grp_df.loc[train_idx,'group']
        val_grps   = grp_df.loc[val_idx,'group']
        df_train = df[df['group'].isin(train_grps)].reset_index(drop=True)
        df_val   = df[df['group'].isin(val_grps)].reset_index(drop=True)

        # 类权重
        cw_labels = le.transform(df_train['class'])
        classes_fold = np.unique(df_train['class_encoded'])
        cw = compute_class_weight(
        class_weight='balanced',
        classes=classes_fold,
        y=cw_labels
        )

        class_weights = dict(zip(classes_fold, cw))

        # val & eval generators
        val_gen = ImageDataGenerator(preprocessing_function=preprocess_input) \
            .flow_from_dataframe(df_val, x_col='filename', y_col='class',
                                 target_size=(260,260), batch_size=batch_size,
                                 class_mode='categorical', shuffle=False, seed=seed)
        eval_train = ImageDataGenerator(preprocessing_function=preprocess_input) \
            .flow_from_dataframe(df_train, x_col='filename', y_col='class',
                                 target_size=(260,260), batch_size=32,
                                 class_mode='categorical', shuffle=False, seed=seed)
        eval_val   = ImageDataGenerator(preprocessing_function=preprocess_input) \
            .flow_from_dataframe(df_val, x_col='filename', y_col='class',
                                 target_size=(260,260), batch_size=32,
                                 class_mode='categorical', shuffle=False, seed=seed)

        # ---- Stage 1 ----
        model = build_model(num_classes=len(np.unique(df['class'])))
        ckpt1 = f"stage1_fold{fold}.keras"
        # 冻结除最后 15 层以外的所有层
        total_layers = len(model.layers)
        for i, layer in enumerate(model.layers):
            layer.trainable = (i >= total_layers-15)
        steps1 = int(np.ceil(len(df_train)/batch_size))
        lr1 = CosineDecay(5e-3, decay_steps=30*steps1, alpha=1e-3)
        model.compile(
            optimizer=Adam(lr1),
            loss=CategoricalCrossentropy(label_smoothing=0.0),
            metrics=['accuracy']
        )
        y1 = pd.get_dummies(df_train['class']).values
        gen1 = AugmentGenerator(
            x_set=df_train['filename'].values,
            y_set=y1,
            batch_size=batch_size,
            alpha=0.4, shuffle=True,
            mode='hybrid', cutmix_prob=0.5,
            seed=seed, mix_ratio=0.4
        )
        cb1 = [
            EarlyStopping('val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(ckpt1, save_best_only=True, verbose=0),
            TensorBoard(log_dir=os.path.join(LOG_ROOT,f"fold{fold}/stage1"))
        ]
        hist1 = model.fit(
            gen1, epochs=30, validation_data=val_gen,
            class_weight=class_weights, callbacks=cb1, verbose=1
        )
        model.save(ckpt1)

        # ---- Stage 2 ----
        model = tf.keras.models.load_model(ckpt1)
        ckpt2 = f"stage2_fold{fold}.keras"
        # 解冻一半层
        total_layers = len(model.layers)
        for i, layer in enumerate(model.layers):
            layer.trainable = (i >= total_layers//2)
        steps2 = int(np.ceil(len(df_train)/batch_size))
        lr2 = CosineDecay(5e-5, decay_steps=20*steps2, alpha=1e-3)
        model.compile(
            optimizer=Adam(lr2),
            loss=CategoricalCrossentropy(label_smoothing=0.05),
            metrics=['accuracy']
        )
        gen2 = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
            shear_range=0.1, zoom_range=0.1, horizontal_flip=True,
            brightness_range=[0.8,1.2], channel_shift_range=20.0,
            fill_mode='nearest'
        ).flow_from_dataframe(
            df_train, x_col='filename', y_col='class',
            target_size=(260,260), batch_size=batch_size,
            class_mode='categorical', shuffle=True, seed=seed
        )
        cb2 = [
            EarlyStopping('val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(ckpt2, save_best_only=True, verbose=0),
            TensorBoard(log_dir=os.path.join(LOG_ROOT,f"fold{fold}/stage2"))
        ]
        hist2 = model.fit(
            gen2, initial_epoch=0, epochs=50,
            validation_data=val_gen,
            class_weight=class_weights, callbacks=cb2, verbose=1
        )
        model.save(ckpt2)

        # ---- Stage 3 ----
        model = tf.keras.models.load_model(ckpt2)
        ckpt3 = f"stage3_fold{fold}.keras"
        for layer in model.layers:
            layer.trainable = True
        steps3 = int(np.ceil(len(df_train)/batch_size))
        lr3 = CosineDecay(5e-6, decay_steps=30*steps3, alpha=1e-3)
        model.compile(
            optimizer=Adam(lr3),
            loss=CategoricalCrossentropy(label_smoothing=0.05),
            metrics=['accuracy']
        )
        gen3 = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
            shear_range=0.1, zoom_range=0.1, horizontal_flip=True,
            brightness_range=[0.9,1.1], channel_shift_range=15.0,
            fill_mode='nearest'
        ).flow_from_dataframe(
            df_train, x_col='filename', y_col='class',
            target_size=(260,260), batch_size=batch_size,
            class_mode='categorical', shuffle=True, seed=seed
        )
        cb3 = [
            EarlyStopping('val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(ckpt3, save_best_only=True, verbose=0),
            TensorBoard(log_dir=os.path.join(LOG_ROOT,f"fold{fold}/stage3"))
        ]
        hist3 = model.fit(
            gen3, initial_epoch=0, epochs=80,
            validation_data=val_gen,
            class_weight=class_weights, callbacks=cb3, verbose=1
        )
        model.save(ckpt3)
        
        

        # 绘制三阶段 loss 曲线
        plot_fold_loss([hist1, hist2, hist3], fold)

        # 最终模型评估（直接用 stage3）
        final_model = tf.keras.models.load_model(ckpt3)
        tr_loss, tr_acc = final_model.evaluate(eval_train, verbose=0)
        vl_loss, vl_acc = final_model.evaluate(eval_val,   verbose=0)
        ts_loss, ts_acc = final_model.evaluate(test_gen,    verbose=0)

        all_train_acc.append(tr_acc)
        all_val_acc.append(vl_acc)
        all_test_acc.append(ts_acc)
        all_train_loss.append(tr_loss)
        all_val_loss.append(vl_loss)
        all_test_loss.append(ts_loss)

    # Cross-fold 汇总
    print("\n=== CV Summary ===")
    print(f"Train Acc: {np.mean(all_train_acc):.4f} ± {np.std(all_train_acc):.4f}")
    print(f"Val   Acc: {np.mean(all_val_acc):.4f} ± {np.std(all_val_acc):.4f}")
    print(f"Test  Acc: {np.mean(all_test_acc):.4f} ± {np.std(all_test_acc):.4f}")
    
    plot_fold_summary({
    'train_acc': all_train_acc,
    'val_acc': all_val_acc,
    'test_acc': all_test_acc,
    'train_loss': all_train_loss,
    'val_loss': all_val_loss,
    'test_loss': all_test_loss,
})