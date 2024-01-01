import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import statistics
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Fungsi untuk membuat heatmap confusion matrix
def plot_confusion_matrix_heatmap(y_true, y_pred, title=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))  # Adjust the size of the plot here
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])

    # Set the title if provided
    if title:
        plt.title(title)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot()

# Fungsi untuk membuat scatter plot K-Nearest Neighbors
def plot_kneighbors_scatter(df, k, features, x1, x2):
    if len(features) != 2:
        st.warning("Pilih tepat dua fitur untuk visualisasi K-Nearest Neighbors.")
        return

    euclidean_distance = []

    for i in range(df.shape[0]):
        dist = np.sqrt(np.dot(df[features].iloc[i].values, df[features].iloc[i].values))
        euclidean_distance.append(dist)

    index = np.argsort(euclidean_distance)
    index = index[:k]
    label = [df.Outcome.iloc[i] for i in index]
    label = statistics.mode(label)

    palette = sns.color_palette("husl", 2)
    colors = {0: palette[0], 1: palette[1]}

    fig, ax = plt.subplots(figsize=(15, 8))
    sns.scatterplot(data=df, x=features[0], y=features[1], hue='Outcome',
                    alpha=0.9, s=250, palette=palette, ax=ax)

    for i in index:
        target_value = df.Outcome.iloc[i]
        if isinstance(target_value, (int, float)):
            color = colors[int(target_value)]
        else:
            color = 'gray'
        ax.scatter(x=df[features[0]].iloc[i], y=df[features[1]].iloc[i], s=250, alpha=0.6, linewidth=2, edgecolor='k', color=color)

    ax.scatter(x=x1, y=x2, s=400, marker='*', color='k')
    ax.set_title(label=f'K-Nearest Neighbor with K = {k}', fontsize=14)
    ax.set_axis_off()
    st.pyplot()

    return f'Predictions: {label}'

# Fungsi untuk membuat ROC Curve
def plot_roc_curve(y_true, y_pred_proba):
    # Plot ROC curve
    plt.figure(figsize=(12, 8))  # Adjust the size of the plot here
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    plt.plot([0, 1], [0, 1], 'g--')
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve')
    st.pyplot()

# Fungsi untuk melatih model k-NN
def train_model(x_train, y_train, n_neighbors=7):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)
    return model

# Fungsi untuk melakukan Grid Search dan plot hasilnya
def plot_grid_search_results(X, y):
    param_grid = {'n_neighbors': np.arange(1, 50)}
    knn = KNeighborsClassifier()
    knn_cv = GridSearchCV(knn, param_grid, cv=5)
    knn_cv.fit(X, y)

    # Plot hasil Grid Search
    plt.figure(figsize=(12, 8))
    plt.plot(knn_cv.cv_results_['param_n_neighbors'], knn_cv.cv_results_['mean_test_score'], marker='o')
    plt.title('Grid Search Results')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Mean Test Score')
    plt.grid(True)
    st.pyplot()  # Menampilkan plot hasil Grid Search

    # Print best score and parameters
    st.write("Best Score:", knn_cv.best_score_)
    st.write("Best Parameters:", knn_cv.best_params_)

# Fungsi untuk membuat plot akurasi KNN Varying
def plot_knn_accuracy(x_train, y_train, x_test, y_test, max_neighbors):
    neighbors = range(1, max_neighbors + 1)
    train_accuracy = []
    test_accuracy = []

    for neighbor in neighbors:
        model = KNeighborsClassifier(n_neighbors=neighbor)
        model.fit(x_train, y_train)
        train_accuracy.append(model.score(x_train, y_train) * 100)
        test_accuracy.append(model.score(x_test, y_test) * 100)

    # Plot k-NN Accuracy
    plt.figure(figsize=(10, 6))
    plt.title('k-NN Varying number of neighbors')
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy', marker='o', linestyle='-', color='orange')
    plt.plot(neighbors, train_accuracy, label='Training accuracy', marker='o', linestyle='-', color='lightblue')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    st.pyplot()

    # Print the accuracy values
    for neighbor, test_acc, train_acc in zip(neighbors, test_accuracy, train_accuracy):
        st.write(f"Neighbors: {neighbor}, Testing Accuracy: {test_acc:.2f}%, Training Accuracy: {train_acc:.2f}")

    # Plot Confusion Matrix for k-NN with max_neighbors
    model_k = KNeighborsClassifier(n_neighbors=max_neighbors)
    model_k.fit(x_train, y_train)
    y_pred_k = model_k.predict(x_test)

    # Set the title for Confusion Matrix
    conf_matrix_title = f'KNN Classifier Confusion Matrix (Neighbors={max_neighbors})'

    # Plot the confusion matrix
    plot_confusion_matrix_heatmap(y_test, y_pred_k, title=conf_matrix_title)

    # Plot ROC Curve for k-NN with max_neighbors
    model = KNeighborsClassifier(n_neighbors=max_neighbors)
    model.fit(x_train, y_train)
    plot_roc_curve(y_test, model.predict_proba(x_test)[:, 1])

    # Print AUC ROC for k-NN with max_neighbors
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)
    y_probs = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test_encoded, y_probs)
    roc_auc = auc(fpr, tpr)
    st.write(f"AUC ROC for k-NN with {max_neighbors} neighbors: {roc_auc:.4f}")

# Fungsi utama aplikasi
def app(df, x, y):
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Halaman Visualisasi Prediksi Diabetes")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Kode untuk mengukur akurasi pada rentang jumlah tetangga dari 1 hingga 9
    test_accuracies = []
    train_accuracies = []

    for n_neighbors in range(1, 10):
        knn = train_model(x_train, y_train, n_neighbors)
        train_accuracies.append(knn.score(x_train, y_train))
        test_accuracies.append(knn.score(x_test, y_test))

    # Generate plots for training and testing accuracy where x is the number of neighbors and y is accuracy
    plt.figure(figsize=(11, 6))
    plt.plot(range(1, 10), train_accuracies, marker='*', label='Train Score')
    plt.plot(range(1, 10), test_accuracies, marker='o', label='Test Score')
    plt.xlabel('Number of neighbors', size='15')
    plt.ylabel('Accuracy', size='15')
    plt.text(7.7, 0.75, 'Here!')
    plt.grid()
    plt.legend()

    if st.checkbox("Plot Akurasi training dan testing"):
        st.pyplot()

    if st.checkbox("Plot Confusion Matrix"):
        st.write("Menggunakan data yang dihasilkan dari model")
        model = train_model(x_train, y_train)
        y_pred = model.predict(x_test)
        plot_confusion_matrix_heatmap(y_test, y_pred)

    if st.checkbox("Plot ROC Curve"):
        st.write("Menggunakan data yang dihasilkan dari model")
        model = train_model(x_train, y_train)

        # Get predicted probabilities for positive class
        y_pred_proba = model.predict_proba(x_test)[:, 1]

        # Plot ROC curve using the function
        plot_roc_curve(y_test, y_pred_proba)

     # Checkbox untuk Plot Grid Search Results
    if st.checkbox("Plot Grid Search Results"):
        st.write("Melakukan Grid Search untuk mencari parameter terbaik")
        plot_grid_search_results(x, y)

    # Checkbox untuk Plot K-Nearest Neighbors
    if st.checkbox("Plot KNN"):
        st.write("Menggunakan implementasi k-NN dengan modul statistics")
        st.write("Kustomisasi plot sesuai")

        # Pilih fitur untuk visualisasi
        feature_options = x.columns.tolist()
        selected_features = st.multiselect('Pilih fitur untuk visualisasi', feature_options, default=[feature_options[0], feature_options[1]])

        # Input nilai K yang diinginkan
        k_value = st.slider('Pilih Nilai K', 1, 20, 3)

        # Check the number of selected features
        if len(selected_features) != 2:
            st.warning("Pilih tepat dua fitur untuk visualisasi K-Nearest Neighbors.")
        else:
            # Input nilai x1 dan x2
            x1_value = st.slider('Nilai value x1', float(x[selected_features[0]].min()), float(x[selected_features[0]].max()), float(x[selected_features[0]].mean()))
            x2_value = st.slider('Nilai value x2', float(x[selected_features[1]].min()), float(x[selected_features[1]].max()), float(x[selected_features[1]].mean()))

            # Setelah semua input diterima, tampilkan plot
            result = plot_kneighbors_scatter(df, k_value, selected_features, x1_value, x2_value)
            st.write(result)

    # Checkbox untuk Plot KNN Accuracy
    if st.checkbox("Plot KNN Accuracy"):
        st.write("Input Nilai K yang Diinginkan")
        max_neighbors = st.slider('Select nilai K of neighbors', 1, 20, 3)
        plot_knn_accuracy(x_train, y_train, x_test, y_test, max_neighbors)

# Panggil fungsi utama aplikasi
if __name__ == '__main__':
    # Assuming these are the columns you want to include
    columns_to_include = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    # Select only the columns you want from the DataFrame
    df = pd.DataFrame()
    x = df[columns_to_include]
    y = df['Outcome']

    app(df, x, y)
