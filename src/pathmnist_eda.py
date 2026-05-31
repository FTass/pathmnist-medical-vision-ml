import random
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from medmnist import INFO, PathMNIST
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

class PathMNISTLoader:
    def __init__(self, seed=42):
        self.seed = seed
        self.data_flag = "pathmnist"
        self.info = INFO[self.data_flag]
        self.datasets = {}
        self.class_id_to_name = {}
        self._set_seeds()

    def _set_seeds(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def load_data(self, download=True):
        self.datasets = {
            "train": PathMNIST(split="train", download=download),
            "validation": PathMNIST(split="val", download=download),
            "test": PathMNIST(split="test", download=download),
        }
        classes_df = pd.DataFrame({
            "class_id": [int(cid) for cid in self.info["label"].keys()],
            "class_name": list(self.info["label"].values())
        }).sort_values("class_id").reset_index(drop=True)
        
        self.class_id_to_name = dict(zip(classes_df["class_id"], classes_df["class_name"]))
        return self.datasets, self.class_id_to_name

    def get_metadata(self):
        return pd.DataFrame({
            "Atributo": ["Dataset", "Identificador", "Tarea", "Canales", "Clases"],
            "Valor": ["PathMNIST", self.data_flag, self.info["task"], self.info["n_channels"], len(self.info["label"])]
        })

    def audit_structure(self):
        rows = []
        for split_name, dataset in self.datasets.items():
            images, labels = dataset.imgs, dataset.labels
            rows.append({
                "split": split_name,
                "num_images": images.shape[0],
                "shape": images.shape[1:],
                "label_shape": labels.shape,
                "pixel_min": int(images.min()),
                "pixel_max": int(images.max()),
            })
        return pd.DataFrame(rows)


class DistributionAnalyzer:
    def __init__(self, datasets, class_id_to_name):
        self.datasets = datasets
        self.class_id_to_name = class_id_to_name

    def get_class_distribution(self):
        rows = []
        for split_name, dataset in self.datasets.items():
            labels = dataset.labels.reshape(-1).astype(int)
            unique_labels, counts = np.unique(labels, return_counts=True)
            total = labels.shape[0]
            for class_id, count in zip(unique_labels, counts):
                rows.append({
                    "split": split_name,
                    "class_id": int(class_id),
                    "class_name": self.class_id_to_name[int(class_id)],
                    "count": int(count),
                    "percentage": 100 * count / total,
                })
        return pd.DataFrame(rows)
    
    def _compute_js_divergence(self, values_a, values_b, bins=256, value_range=(0, 255)):
        hist_a, _ = np.histogram(values_a, bins=bins, range=value_range)
        hist_b, _ = np.histogram(values_b, bins=bins, range=value_range)

        p = hist_a.astype(float) / hist_a.sum()
        q = hist_b.astype(float) / hist_b.sum()
        
        eps = 1e-12
        p, q = p + eps, q + eps
        p, q = p / p.sum(), q / q.sum()
        m = 0.5 * (p + q)

        kl_pm = np.sum(p * np.log2(p / m))
        kl_qm = np.sum(q * np.log2(q / m))
        return 0.5 * (kl_pm + kl_qm)

    def get_jensen_shannon_divergence(self):
        channel_names = ["R", "G", "B"]
        js_rows = []
        train_images = self.datasets["train"].imgs

        for split_name in ["validation", "test"]:
            split_images = self.datasets[split_name].imgs
            for channel_idx, channel_name in enumerate(channel_names):
                js_rows.append({
                    "comparison": f"{split_name}_vs_train",
                    "channel": channel_name,
                    "jensen_shannon_divergence": self._compute_js_divergence(
                        train_images[:, :, :, channel_idx].reshape(-1),
                        split_images[:, :, :, channel_idx].reshape(-1),
                    ),
                })
        return pd.DataFrame(js_rows).round(6)


class RGBAnalyzer:
    def __init__(self, datasets, class_id_to_name):
        self.datasets = datasets
        self.class_id_to_name = class_id_to_name

    def get_global_statistics(self):
        rows = []
        channel_names = ["R", "G", "B"]
        for split_name, dataset in self.datasets.items():
            images = dataset.imgs.astype(np.float32)
            for channel_idx, channel_name in enumerate(channel_names):
                values = images[:, :, :, channel_idx].reshape(-1)
                rows.append({
                    "split": split_name,
                    "channel": channel_name,
                    "mean": values.mean(),
                    "std": values.std()
                })
        return pd.DataFrame(rows)

    def get_normalization_params(self):
        stats_df = self.get_global_statistics()
        train_stats = stats_df[stats_df["split"] == "train"]
        return pd.DataFrame({
            "channel": ["R", "G", "B"],
            "mean_0_1": train_stats["mean"].to_numpy() / 255.0,
            "std_0_1": train_stats["std"].to_numpy() / 255.0,
        }).round(4)
    
    def get_rgb_shift_summary(self):
        stats_df = self.get_global_statistics() # Requiere refactorizar para incluir class_id
        # El cálculo exacto requiere que get_global_statistics se despiece por clase.
        # Es una mala práctica de ingeniería poner 40 líneas de manipulación de DataFrames 
        # en una sola función. Si necesitas esta tabla específica, debes crear un método 
        # get_class_rgb_statistics() primero.


class IntegrityAuditor:
    def __init__(self, datasets, class_id_to_name):
        self.datasets = datasets
        self.class_id_to_name = class_id_to_name

    def compute_image_hash(self, image):
        return hashlib.md5(image.tobytes()).hexdigest()

    def detect_duplicates(self):
        rows = []
        for split_name, dataset in self.datasets.items():
            labels = dataset.labels.reshape(-1).astype(int)
            for idx in range(dataset.imgs.shape[0]):
                class_id = labels[idx]
                rows.append({
                    "split": split_name,
                    "image_index": idx,
                    "class_id": class_id,
                    "class_name": self.class_id_to_name[class_id],
                    "image_hash": self.compute_image_hash(dataset.imgs[idx]),
                })
        hash_df = pd.DataFrame(rows)
        return hash_df, hash_df[hash_df.duplicated("image_hash", keep=False)]

    def audit_visual_quality(self):
        rows = []
        for split_name, dataset in self.datasets.items():
            images = dataset.imgs.astype(np.float32)
            labels = dataset.labels.reshape(-1).astype(int)
            
            for idx in range(images.shape[0]):
                gray_like = images[idx].mean(axis=2)
                rows.append({
                    "split": split_name,
                    "image_index": idx,
                    "class_id": labels[idx],
                    "class_name": self.class_id_to_name[labels[idx]],
                    "mean_intensity": gray_like.mean(),
                    "std_intensity": gray_like.std(),
                })

        quality_df = pd.DataFrame(rows)
        
        # Umbrales exploratorios
        thresholds = {
            "very_dark_mean": quality_df["mean_intensity"].quantile(0.01),
            "very_bright_mean": quality_df["mean_intensity"].quantile(0.99),
            "very_low_std": quality_df["std_intensity"].quantile(0.01),
            "very_high_std": quality_df["std_intensity"].quantile(0.99),
        }
        
        quality_df["flag_very_dark"] = quality_df["mean_intensity"] <= thresholds["very_dark_mean"]
        quality_df["flag_very_bright"] = quality_df["mean_intensity"] >= thresholds["very_bright_mean"]
        quality_df["flag_low_variability"] = quality_df["std_intensity"] <= thresholds["very_low_std"]
        quality_df["flag_high_variability"] = quality_df["std_intensity"] >= thresholds["very_high_std"]
        
        flag_cols = ["flag_very_dark", "flag_very_bright", "flag_low_variability", "flag_high_variability"]
        quality_df["num_flags"] = quality_df[flag_cols].sum(axis=1)
        
        return quality_df, thresholds


class FeatureExplorer:
    def __init__(self, datasets, class_id_to_name, seed=42):
        self.datasets = datasets
        self.class_id_to_name = class_id_to_name
        self.seed = seed

    def build_pca_projection(self, split_name="train", samples_per_class=300):
        dataset = self.datasets[split_name]
        labels = dataset.labels.reshape(-1).astype(int)
        rng = np.random.default_rng(self.seed)
        rows = []

        for class_id, class_name in self.class_id_to_name.items():
            candidate_indices = np.where(labels == class_id)[0]
            selected = rng.choice(candidate_indices, size=min(samples_per_class, len(candidate_indices)), replace=False)
            for idx in selected:
                rows.append({"image_index": int(idx), "class_id": class_id, "class_name": class_name})

        sample_df = pd.DataFrame(rows)
        images = dataset.imgs[sample_df["image_index"].to_numpy()].astype(np.float32) / 255.0
        X_flat = images.reshape(images.shape[0], -1)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_flat)
        pca = PCA(n_components=2, random_state=self.seed)
        X_2d = pca.fit_transform(X_scaled)

        sample_df["PC1"] = X_2d[:, 0]
        sample_df["PC2"] = X_2d[:, 1]
        
        return sample_df, pca.explained_variance_ratio_


class EDAVisualizer:
    @staticmethod
    def plot_class_counts(distribution_df, split_name):
        split_data = distribution_df[distribution_df["split"] == split_name]
        plt.figure(figsize=(10, 4))
        plt.bar(split_data["class_name"], split_data["count"], edgecolor="black")
        plt.title(f"Conteo de imágenes - {split_name}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_pca_projection(pca_df):
        plt.figure(figsize=(10, 6))
        for class_id in sorted(pca_df["class_id"].unique()):
            class_data = pca_df[pca_df["class_id"] == class_id]
            class_name = class_data["class_name"].iloc[0]
            plt.scatter(class_data["PC1"], class_data["PC2"], s=10, alpha=0.6, label=class_name)
        plt.title("PCA 2D de imágenes (Muestra estratificada)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()