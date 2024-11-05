# MT-vocab-tuning

Reserach aiming to understand influence of tunned vocabulary on low-resource language learning via MT problems


### Dataset installation

If you are interested to download all the data, we provide a pipeline to do that. Just open `/notebooks` folder and you'll find 3 files:
* `/notebooks/load_kazakh_datasets.ipynb`
* `/notebooks/load_mari_datasets.ipynb`
* `/notebooks/load_tatar_datasets.ipynb`

Just run required cells (for more info read markdown provided) and datasets will load to `/datasets/{lang}` folder for monolingual data, and `/datasets/{src}-{dst}` for parallel corpora.


### Evaluating raw models

**Madlad**

```commandline
========== C:\\Users\\cutefluffyfox\\PycharmProjects\\MT-vocab-tuning\\data\\benchmark\\mhr-ru\\flores-dev-float16.txt ==========
bleu : 0.1434093052494062
bleurt : 0.01981488582766881
bertscore : 0.7639593007332582
chrf : 37.90391931929389
google_bleu : 0.1691604913206781
========== C:\\Users\\cutefluffyfox\\PycharmProjects\\MT-vocab-tuning\\data\\benchmark\\mhr-ru\\flores-dev-float32.txt ==========
bleu : 0.14384333535033544
bleurt : 0.020304277565408733
bertscore : 0.7640489077974585
chrf : 37.90642608569875
google_bleu : 0.16957811135260303

========== C:\\Users\\cutefluffyfox\\PycharmProjects\\MT-vocab-tuning\\data\\benchmark\\kaz-ru\\flores-dev-float16.txt ==========
bleu : 0.18082925248444684
bleurt : 0.09459315444022387
bertscore : 0.7980259766310842
chrf : 43.88523331699426
google_bleu : 0.21029838513169263
========== C:\\Users\\cutefluffyfox\\PycharmProjects\\MT-vocab-tuning\\data\\benchmark\\kaz-ru\\flores-dev-float32.txt ==========
bleu : 0.18065186902901745
bleurt : 0.09470855741721458
bertscore : 0.7980790147212229
chrf : 43.887438154016564
google_bleu : 0.2103576734594319

========== C:\\Users\\cutefluffyfox\\PycharmProjects\\MT-vocab-tuning\\data\\benchmark\\tat-ru\\flores-dev-float16.txt ==========
bleu : 0.0741314385850048
bleurt : -0.11265124618229919
bertscore : 0.7066054013571744
chrf : 25.203203336362336
google_bleu : 0.10190579849328808
```

