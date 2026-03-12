import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

data_dir = 'dataset'

print("Script started")
print("Dataset folders:", os.listdir(data_dir))

hog_features = []
hog_labels = []
plt.figure(figsize=(10,5))
display_count = 0

for label in os.listdir(data_dir):
    print("Processing folder:", label)
    for file in os.listdir(os.path.join(data_dir, label)):
        print("Reading image:", file)
        img_path = os.path.join(data_dir, label, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128,128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features, hog_img = hog(gray, orientations=9,
                                pixels_per_cell=(8,8),
                                cells_per_block=(2,2),
                                visualize=True)
        hog_features.append(features)
        hog_labels.append(label)

        if display_count < 3:
            plt.subplot(2,3,display_count+1)
            plt.imshow(gray, cmap='gray')
            plt.title('Original')
            plt.axis('off')
            plt.subplot(2,3,display_count+4)
            plt.imshow(hog_img, cmap='gray')
            plt.title('HOG')
            plt.axis('off')
            display_count += 1

plt.tight_layout()
plt.show()

hog_df = pd.DataFrame(hog_features)
hog_df['label'] = hog_labels
hog_df.to_csv('hog_features.csv', index=False)
hog_df.head()

X = hog_df.drop('label', axis=1).values
y = hog_df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_hog = SVC(kernel='rbf', probability=True)
svm_hog.fit(X_train, y_train)

y_pred = svm_hog.predict(X_test)
print('HOG Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix - HOG')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

y_prob = svm_hog.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve((y_test=='dogs').astype(int), y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr)
plt.title(f'ROC Curve (AUC = {roc_auc:.2f})')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = ImageFolder(data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

cnn_features = []
cnn_labels = []

with torch.no_grad():
    for imgs, lbls in loader:
        output = model(imgs)
        output = output.view(output.size(0), -1)
        cnn_features.append(output.numpy())
        cnn_labels.append(lbls.numpy())

X_cnn = np.vstack(cnn_features)
y_cnn = np.hstack(cnn_labels)

X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    X_cnn, y_cnn, test_size=0.2, random_state=42)

svm_cnn = SVC(kernel='rbf')
svm_cnn.fit(X_train_cnn, y_train_cnn)

y_pred_cnn = svm_cnn.predict(X_test_cnn)
print('CNN Feature Accuracy:', accuracy_score(y_test_cnn, y_pred_cnn))

print('HOG Accuracy:', accuracy_score(y_test, y_pred))
print('CNN Accuracy:', accuracy_score(y_test_cnn, y_pred_cnn))