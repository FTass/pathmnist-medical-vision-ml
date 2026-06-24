# src/eda.py

import numpy as np
import pandas as pd
import hashlib
import matplotlib.pyplot as plt
from medmnist import INFO, PathMNIST
from typing import Dict, Tuple

class PathMNISTEDA:
    """EDA completo de PathMNIST en una sola clase."""
    
    def __init__(self, data_flag="pathmnist", seed=42):
        self.data_flag = data_flag
        self.seed = seed
        np.random.seed(seed)
        
        self.info = INFO[data_flag]
        self.datasets = {}
        self.class_id_to_name = {}
        self.distribution_df = None
    
    # ============= CARGA =============
    def load(self):
        """Carga los tres splits."""
        self.datasets = {
            "train": PathMNIST(split="train", download=True),
            "validation": PathMNIST(split="val", download=True),
            "test": PathMNIST(split="test", download=True),
        }
        
        # Mapeo de clases
        class_labels = self.info["label"]
        self.class_id_to_name = {int(cid): name for cid, name in class_labels.items()}
        
        return self
    
    # ============= METADATOS =============
    def metadata_table(self) -> pd.DataFrame:
        """Tabla de metadatos del dataset."""
        return pd.DataFrame({
            "Atributo": ["Dataset", "Tarea", "Canales", "Clases", "Splits"],
            "Valor": [
                self.info["python_class"],
                self.info["task"],
                self.info["n_channels"],
                len(self.info["label"]),
                list(self.info["n_samples"].keys()),
            ]
        })
    
    # ============= DISTRIBUCIÓN =============
    def distribution_summary(self) -> pd.DataFrame:
        """Resumen de distribución de clases por split."""
        rows = []
        
        for split_name, dataset in self.datasets.items():
            labels = dataset.labels.reshape(-1).astype(int)
            unique, counts = np.unique(labels, return_counts=True)
            
            for class_id, count in zip(unique, counts):
                rows.append({
                    "split": split_name,
                    "class_id": int(class_id),
                    "class_name": self.class_id_to_name[int(class_id)],
                    "count": int(count),
                    "percentage": 100 * count / len(labels),
                })
        
        self.distribution_df = pd.DataFrame(rows)
        return self.distribution_df
    
    def imbalance_report(self) -> pd.DataFrame:
        """Reporte de desbalance por split."""
        if self.distribution_df is None:
            self.distribution_summary()
        
        return self.distribution_df.groupby("split").agg(
            min_count=("count", "min"),
            max_count=("count", "max"),
            mean_count=("count", "mean"),
        ).reset_index().assign(
            max_to_min_ratio=lambda df: (df["max_count"] / df["min_count"]).round(3)
        )
    
    # ============= ESTADÍSTICAS RGB =============
    def rgb_statistics(self) -> pd.DataFrame:
        """Media y std RGB por split."""
        rows = []
        
        for split_name, dataset in self.datasets.items():
            images = dataset.imgs.astype(np.float32)
            for ch_idx, ch_name in enumerate(["R", "G", "B"]):
                values = images[:, :, :, ch_idx].reshape(-1)
                rows.append({
                    "split": split_name,
                    "channel": ch_name,
                    "mean": values.mean(),
                    "std": values.std(),
                    "min": values.min(),
                    "max": values.max(),
                })
        
        return pd.DataFrame(rows)
    
    def normalization_params(self) -> Dict:
        """Parámetros de normalización (train only)."""
        rgb_stats = self.rgb_statistics()
        train = rgb_stats[rgb_stats["split"] == "train"]
        
        params = {}
        for _, row in train.iterrows():
            ch = row["channel"]
            params[ch] = {
                "mean_0_255": row["mean"],
                "std_0_255": row["std"],
                "mean_0_1": row["mean"] / 255.0,
                "std_0_1": row["std"] / 255.0,
            }
        
        return params
    
    # ============= AUDITORÍA DE CALIDAD =============
    def quality_audit(self) -> Tuple[pd.DataFrame, Dict]:
        """Detecta imágenes extremas por intensidad."""
        rows = []
        
        for split_name, dataset in self.datasets.items():
            images = dataset.imgs.astype(np.float32)
            labels = dataset.labels.reshape(-1).astype(int)
            
            for idx in range(images.shape[0]):
                gray = images[idx].mean(axis=2)
                rows.append({
                    "split": split_name,
                    "image_index": idx,
                    "class_name": self.class_id_to_name[labels[idx]],
                    "mean_intensity": gray.mean(),
                    "std_intensity": gray.std(),
                })
        
        quality_df = pd.DataFrame(rows)
        
        # Thresholds
        thresholds = {
            "very_dark": quality_df["mean_intensity"].quantile(0.01),
            "very_bright": quality_df["mean_intensity"].quantile(0.99),
            "low_var": quality_df["std_intensity"].quantile(0.01),
            "high_var": quality_df["std_intensity"].quantile(0.99),
        }
        
        # Flags
        quality_df["flag_dark"] = quality_df["mean_intensity"] <= thresholds["very_dark"]
        quality_df["flag_bright"] = quality_df["mean_intensity"] >= thresholds["very_bright"]
        quality_df["flag_low_var"] = quality_df["std_intensity"] <= thresholds["low_var"]
        quality_df["flag_high_var"] = quality_df["std_intensity"] >= thresholds["high_var"]
        quality_df["num_flags"] = (quality_df["flag_dark"] + quality_df["flag_bright"] + 
                                   quality_df["flag_low_var"] + quality_df["flag_high_var"]).astype(int)
        
        return quality_df, thresholds
    
    # ============= DUPLICADOS =============
    def leakage_detection(self) -> Dict:
        """Detecta leakage entre splits mediante hashing."""
        hash_dict = {}
        
        for split_name, dataset in self.datasets.items():
            hashes = set()
            for idx in range(dataset.imgs.shape[0]):
                image_hash = hashlib.md5(dataset.imgs[idx].tobytes()).hexdigest()
                hashes.add(image_hash)
            hash_dict[split_name] = hashes
        
        # Comparar splits
        leakage = {}
        for split_a, split_b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            shared = hash_dict[split_a].intersection(hash_dict[split_b])
            leakage[f"{split_a}_vs_{split_b}"] = {
                "shared_hashes": len(shared),
                "has_leakage": len(shared) > 0,
            }
        
        return leakage
    
    # ============= VISUALIZACIONES =============
    def plot_distribution(self, split_name="train"):
        """Gráfico de conteo por clase."""
        if self.distribution_df is None:
            self.distribution_summary()
        
        split_data = self.distribution_df[self.distribution_df["split"] == split_name]
        
        plt.figure(figsize=(12, 5))
        plt.bar(split_data["class_name"], split_data["count"], edgecolor="black")
        plt.title(f"Conteo de imágenes - {split_name}")
        plt.xlabel("Clase")
        plt.ylabel("Número de imágenes")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    
    def plot_rgb_stats(self, stat="mean"):
        """Gráfico de estadísticas RGB."""
        rgb_stats = self.rgb_statistics()
        pivot = rgb_stats.pivot(index="split", columns="channel", values=stat).reset_index()
        
        x = np.arange(len(pivot))
        width = 0.25
        
        plt.figure(figsize=(8, 5))
        plt.bar(x - width, pivot["R"], width, label="R", edgecolor="black")
        plt.bar(x, pivot["G"], width, label="G", edgecolor="black")
        plt.bar(x + width, pivot["B"], width, label="B", edgecolor="black")
        plt.title(f"{'Media' if stat == 'mean' else 'Desviación'} RGB")
        plt.xlabel("Split")
        plt.xticks(x, pivot["split"])
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_samples(self, n_classes_show=None):
        """Muestra muestras por clase y split."""
        if n_classes_show is None:
            n_classes_show = len(self.class_id_to_name)
        
        class_ids = list(self.class_id_to_name.keys())[:n_classes_show]
        split_names = list(self.datasets.keys())
        
        fig, axes = plt.subplots(len(class_ids), len(split_names), figsize=(10, 2*len(class_ids)))
        
        for row, class_id in enumerate(class_ids):
            for col, split_name in enumerate(split_names):
                dataset = self.datasets[split_name]
                labels = dataset.labels.reshape(-1).astype(int)
                indices = np.where(labels == class_id)[0]
                
                if len(indices) > 0:
                    idx = np.random.choice(indices)
                    ax = axes[row, col]
                    ax.imshow(dataset.imgs[idx])
                    ax.axis("off")
                    
                    if row == 0:
                        ax.set_title(split_name)
                    if col == 0:
                        ax.set_ylabel(self.class_id_to_name[class_id], rotation=0, ha="right")
        
        plt.tight_layout()
        plt.show()
    
    # ============= REPORTE FINAL =============
    def full_report(self):
        """Ejecuta todas las fases y retorna resumen."""
        print("\n" + "="*60)
        print("PathMNIST EDA - REPORTE COMPLETO")
        print("="*60)
        
        print("\nMETADATOS")
        print(self.metadata_table())
        
        print("\nDISTRIBUCIÓN")
        print(self.distribution_summary())
        print("\nDESBALANCE")
        print(self.imbalance_report())
        
        print("\nESTADÍSTICAS RGB")
        print(self.rgb_statistics())
        print("\nNORMALIZACIÓN (train)")
        params = self.normalization_params()
        for ch, vals in params.items():
            print(f"{ch}: mean={vals['mean_0_1']:.4f}, std={vals['std_0_1']:.4f}")
        
        print("\nCALIDAD")
        quality_df, _ = self.quality_audit()
        flagged_count = (quality_df["num_flags"] > 0).sum()
        print(f"Total imágenes con flags: {flagged_count}")
        
        print("\nLEAKAGE")
        leakage = self.leakage_detection()
        for comp, result in leakage.items():
            status = "LEAKAGE" if result["has_leakage"] else "✓ OK"
            print(f"{comp}: {result['shared_hashes']} duplicados - {status}")
        
        print("\n" + "="*60)
        return {
            "distribution": self.distribution_df,
            "quality": quality_df,
            "leakage": leakage,
            "normalization": params,
        }