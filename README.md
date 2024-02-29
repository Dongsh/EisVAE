# EisVAE

Deep clustering in subglacial radar reflectance reveals subglacial lakes


Python (>=3.6) environment is required. You can install all the required python packages by

```
pip install -r requirements.txt
```

Each folder corresponds to different steps of the process:

1. `pick_train_data_from_radar_images/pick_data_from_IPR_images.py`: Pick the ice bottom reflections from a set of radar images (From CReSIS dataset).
2. `VAE_train/vae_main.py`: Train Variational Auto-Encoder (VAE) from the picked data set of radar reflections.
3. `clustering_k-means/clustering_by_k-means.py`: Apply K-means clustering on the encoded radar reflections.
4. `detecting_lakes_by_models/EisVAE_detectors.py`: Use the trained encoder and saved K-means model to detect subglacial lakes from a set of radar images.

Demonstration data and models were contained. You can run each step separately.



For each part of the code (in different folders), run it in its folder separately:

```
cd ${folder}
python ${filename.py}
```
