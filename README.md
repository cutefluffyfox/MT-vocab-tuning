# MT-vocab-tuning

Reserach aiming to understand influence of tunned vocabulary on low-resource language learning via MT problems


### Dataset installation

If you are interested to download all the data, we provide a pipeline to do that. Just open `/notebooks` folder and you'll find 3 files:
* `/notebooks/load_kazakh_datasets.ipynb`
* `/notebooks/load_mari_datasets.ipynb`
* `/notebooks/load_tatar_datasets.ipynb`

Just run required cells (for more info read markdown provided) and datasets will load to `/datasets/{lang}` folder for monolingual data, and `/datasets/{src}-{dst}` for parallel corpora.


