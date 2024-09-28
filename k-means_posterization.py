from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import distance

import streamlit as st
from PIL import Image


@dataclass
class ImageKMeans:
    n_clusters: int = 5
    init: str | np.ndarray = "random"
    max_iter: int = 100
    random_state: int = 42

    def fit(self, image: np.ndarray) -> ImageKMeans:
        """Fit the model to the image"""
        # convert image to list of RGB values
        X = image.reshape(-1, 3)

        # select N random samples as initial centroids
        self._init_centroids(X)

        # iterate until convergence
        for _ in range(self.max_iter):
            # assign each sample to the closest centroid
            y = self._assign_centroids(X)
            self._update_centroids(X, y)

        return self

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return the labels of the image"""
        # convert image to list of RGB values
        X = self._image_as_array(image)

        # assign each sample to the closest centroid
        labels = self._assign_centroids(X)

        # convert labels to matrix
        return labels.reshape(image.shape[:2])

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Return the compressed image"""
        labels = self.predict(image)
        image_compressed = self.centroids_[labels.astype(int)]
        return image_compressed.astype(int)

    def _image_as_array(self, image: np.ndarray) -> np.ndarray:
        """Convert image to list of RGB values"""
        return image.reshape(-1, 3).astype(int)

    def _init_centroids(self, X: np.ndarray) -> None:
        """Select N random samples as initial centroids"""
        if isinstance(self.init, str):
            if self.init == "random":
                # set random seed
                np.random.seed(self.random_state)

                # select N random samples as initial centroids
                mask = np.random.choice(len(X), self.n_clusters, replace=False)
                self.centroids_ = X[mask].copy()
            else:
                raise ValueError(f"Unrecognized str init: {self.init}")
        elif isinstance(self.init, np.ndarray):
            # check that init has the correct shape
            shape_expected = (self.n_clusters, 3)
            if self.init.shape != shape_expected:
                shape = self.init.shape
                msg = f"Expected init shape {shape_expected}, got: {shape}"
                raise ValueError(msg)

            # check that init has the correct values
            if np.any(self.init.flatten() < 0):
                msg = "Expected init values to be in [0, 255]"
                raise ValueError(msg)

            if np.any(self.init.flatten() > 255):
                msg = "Expected init values to be in [0, 255]"
                raise ValueError(msg)

            # check that init has unique values
            if len(np.unique(self.init, axis=0)) != self.n_clusters:
                msg = "Expected init to have unique values"
                raise ValueError(msg)

            self.centroids_ = self.init.copy()
        else:
            raise TypeError(f"Unrecognized init type: {self.init}")

    def _update_centroids(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update the centroids by taking the mean of its samples"""
        for i in range(self.n_clusters):
            centroid = np.mean(X[y == i], axis=0).astype(int)
            self.centroids_[i] = centroid

    def _assign_centroids(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample to the closest centroid"""
        # compute the distance between each sample and each centroid
        dist = distance.cdist(X, self.centroids_, metric="euclidean")

        # assign each sample to the closest centroid
        y = np.argmin(dist, axis=1)

        return y


@dataclass
class ImageKMedoids:
    n_clusters: int = 5
    init: str | np.ndarray = "random"
    max_iter: int = 100
    random_state: int = 42

    def fit(self, image: np.ndarray) -> ImageKMedoids:
        """Fit the model to the image"""
        # convert image to list of RGB values
        X = image.reshape(-1, 3)

        # select N random samples as initial centroids
        self._init_centroids(X)

        # iterate until convergence
        for _ in range(self.max_iter):
            # assign each sample to the closest centroid
            y = self._assign_centroids(X)
            self._update_centroids(X, y)

        return self

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return the labels of the image"""
        # convert image to list of RGB values
        X = self._image_as_array(image)

        # assign each sample to the closest centroid
        labels = self._assign_centroids(X)

        # convert labels to matrix
        return labels.reshape(image.shape[:2])

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Return the compressed image"""
        labels = self.predict(image)
        image_compressed = self.centroids_[labels.astype(int)]
        return image_compressed

    def _image_as_array(self, image: np.ndarray) -> np.ndarray:
        """Convert image to list of RGB values"""
        return image.reshape(-1, 3)

    def _init_centroids(self, X: np.ndarray) -> None:
        """Select N random samples as initial centroids"""
        if isinstance(self.init, str):
            if self.init == "random":
                # set random seed
                np.random.seed(self.random_state)

                # select N random samples as initial centroids
                mask = np.random.choice(len(X), self.n_clusters, replace=False)
                self.centroids_ = X[mask].copy()
            else:
                raise ValueError(f"Unrecognized str init: {self.init}")
        elif isinstance(self.init, np.ndarray):
            # check that init has the correct shape
            shape_expected = (self.n_clusters, 3)
            if self.init.shape != shape_expected:
                shape = self.init.shape
                msg = f"Expected init shape {shape_expected}, got: {shape}"
                raise ValueError(msg)

            # check that init has the correct values
            if np.any(self.init.flatten() < 0):
                msg = "Expected init values to be in [0, 255]"
                raise ValueError(msg)

            if np.any(self.init.flatten() > 255):
                msg = "Expected init values to be in [0, 255]"
                raise ValueError(msg)

            # check that init has unique values
            if len(np.unique(self.init, axis=0)) != self.n_clusters:
                msg = "Expected init to have unique values"
                raise ValueError(msg)

            self.centroids_ = self.init.copy()
        else:
            raise TypeError(f"Unrecognized init type: {self.init}")

    def _update_centroids(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update the centroids by taking the mean of its samples"""
        for i in range(self.n_clusters):
            cluster_points = X[y == i]
            pairwise_distances_matrix = distance.cdist(cluster_points, cluster_points, 
                                                       metric="cityblock")
            # sum of L1 norms for each point
            sum_distances_of_i = np.sum(pairwise_distances_matrix, axis=0)
            min_dist_index = np.argmin(sum_distances_of_i)
            self.centroids_[i] = cluster_points[min_dist_index]

    def _assign_centroids(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample to the closest centroid"""
        # compute the distance between each sample and each centroid
        dist = distance.cdist(X, self.centroids_, metric="cityblock")

        # assign each sample to the closest centroid
        y = np.argmin(dist, axis=1)

        return y


@dataclass
class ImageKMedians:
    n_clusters: int = 5
    init: str | np.ndarray = "random"
    max_iter: int = 100
    random_state: int = 42

    def fit(self, image: np.ndarray) -> ImageKMedians:
        """Fit the model to the image"""
        # convert image to list of RGB values
        X = image.reshape(-1, 3)

        # select N random samples as initial centroids
        self._init_centroids(X)

        # iterate until convergence
        for _ in range(self.max_iter):
            # assign each sample to the closest centroid
            y = self._assign_centroids(X)
            self._update_centroids(X, y)

        return self

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return the labels of the image"""
        # convert image to list of RGB values
        X = self._image_as_array(image)

        # assign each sample to the closest centroid
        labels = self._assign_centroids(X)

        # convert labels to matrix
        return labels.reshape(image.shape[:2])

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Return the compressed image"""
        labels = self.predict(image)
        image_compressed = self.centroids_[labels.astype(int)]
        return image_compressed.astype(int)

    def _image_as_array(self, image: np.ndarray) -> np.ndarray:
        """Convert image to list of RGB values"""
        return image.reshape(-1, 3).astype(np.uint8)

    def _init_centroids(self, X: np.ndarray) -> None:
        """Select N random samples as initial centroids"""
        if isinstance(self.init, str):
            if self.init == "random":
                # set random seed
                np.random.seed(self.random_state)

                # select N random samples as initial centroids
                mask = np.random.choice(len(X), self.n_clusters, replace=False)
                self.centroids_ = X[mask].copy()
            else:
                raise ValueError(f"Unrecognized str init: {self.init}")
        elif isinstance(self.init, np.ndarray):
            # check that init has the correct shape
            shape_expected = (self.n_clusters, 3)
            if self.init.shape != shape_expected:
                shape = self.init.shape
                msg = f"Expected init shape {shape_expected}, got: {shape}"
                raise ValueError(msg)

            # check that init has the correct values
            if np.any(self.init.flatten() < 0):
                msg = "Expected init values to be in [0, 255]"
                raise ValueError(msg)

            if np.any(self.init.flatten() > 255):
                msg = "Expected init values to be in [0, 255]"
                raise ValueError(msg)

            # check that init has unique values
            if len(np.unique(self.init, axis=0)) != self.n_clusters:
                msg = "Expected init to have unique values"
                raise ValueError(msg)

            self.centroids_ = self.init.copy()
        else:
            raise TypeError(f"Unrecognized init type: {self.init}")

    def _update_centroids(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update the centroids by taking the median of its samples"""
        for i in range(self.n_clusters):
            centroid = np.median(X[y == i], axis=0).astype(int)
            self.centroids_[i] = centroid

    def _assign_centroids(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample to the closest centroid, using manhatten distance (L1)"""
        # compute the distance between each sample and each centroid
        dist = distance.cdist(X, self.centroids_, metric="cityblock")

        # assign each sample to the closest centroid
        y = np.argmin(dist, axis=1)

        return y



def run():
    from skimage.data import coffee
    
    # Кнопка для загрузки изображения
    uploaded_image = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

    # Первый ползунок для выбора числа от 0 до 256
    n_clusters = st.slider('Выберите количество цветов в сжатом изображении', 1, 256)

    # Второй ползунок для выбора одного из трех значений
    algorithm = st.selectbox('Выберите алгоритм', ['k-means', 'k-medians', 'k-medoids'])


    # Кнопка "Transform" для получения сжатого изображения
    if st.button('Transform') and uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        # Проверка и преобразование изображения в RGB, если есть альфа-канал
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image = np.array(image)
        
        max_iter = 100
        if algorithm == "k-means":
            model = ImageKMeans(n_clusters=n_clusters, max_iter=max_iter)
        elif algorithm == "k-medians":
            model = ImageKMedians(n_clusters=n_clusters, max_iter=max_iter)
        elif algorithm == "k-medoids":
            model = ImageKMedoids(n_clusters=n_clusters, max_iter=max_iter)

        model.fit(image)

        image_compressed = model.transform(image)
        
        st.image(image_compressed, caption='Сжатое изображение', use_column_width=True)

    # Вывод выбранных значений
    st.write('Выбрано число:', n_clusters)
    st.write('Выбранный алгоритм:', algorithm)


if __name__ == "__main__":
    run()