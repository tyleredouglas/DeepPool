# DeepPool

DeepPool (formerly Haplomatic) is a deep-learning-based tool for improving the localization of quantitative trait loci (QTL) from pooled-sequencing data using haplotype composition. While narrower QTL peaks improve downstream identification of candidate genes at causal loci, inappropriately high resolution during QTL scans can introduce noise and reduce QTL detection power. DeepPool addresses this trade-off between mapping resolution and estimation error by  predicting haplotype composition estimation error and adjusting resolution until predicted error falls below a user-defined threshold, resulting in better localization of genetic signals without sacrificing accuracy. Error predictions are made by hybrid neural network that combines a transformer-encoder that processes a matrix of all SNPs in a given genomic window, and a feed-forward regressor that processes summary features of local genomic characteristics. DeepPool has been validated on both simulated and empirical datsets.


## Features
- Simulation of read-data from in silico populations experiencing recombination and drift to create training datasets
- Quantificaiotn of haplotype composition via Bayesian MCMC.
- Deep learning-based prediction of haplotype composition estimation error.
- Modules for feature generation, model training, fine-tuning, and validation.

## Installation

```bash
git clone https://github.com/tyleredouglas/DeepPool.git
cd DeepPool
pip install -r requirements.txt
```

## Commands

### dp-sim

Simulates pooled population data from in silico populaitons for model training and validation.

**Required Files**

- `--haplotypes` (list of haplotype names, one per line)  
- `--regions` (list of target regions: contig:start:end)
- `--founder-fastas` (one FASTA per contig each with sequences matching haplotype IDs)
- `--contigs` list of contig names (must match FASTA and RILs)
  
**Example:**

```bash
dp-sim \
  --haplotypes hap_names.txt # list of haplotype names
  --ril-df RILS.csv # table of RILs 
  --n 300 \ # population size
  --generations 10 \
  --recomb-rate .5 \
  --n-sims 5 \
  --coverage 100 \
  --read-length 150 \
  --founder-fastas B.3L.fasta \
  --output-dir where/to/save \
  --contigs chr3L \
  --regions regions.txt \
  --threads 16 
```

### dp-window

Generates genomic windows and extracts features for model training.

**Required Files**
- `--populations` (text file listing population names, one per line)
- `--haplotypes-file` (text file listing haplotype names, one per line)
- `--snp-freqs` (CSV with columns: chrom, pos, founder frequencies, simulated frequencies)
- `--true-freq-dir` (directory containing `<pop>_true_freqs.csv` files from `dp-sim`)
- `--output` (root name for output CSV of calculated features)

**Example:**

```bash
dp-window \
  --populations     sim1_5.txt \ # populations to calculate features from
  --founders-file   hap_names.txt \ # list of haplotype names
  --snp-freqs       path/to/snp_freqs_file \ # table of table of snp frequencies in founders and simulated populations (see walkthrough)
  --true-freq-dir   path/to/true_freqs_directory \
  --window-sizes-kb 30 60 70 80 90 150 250 \ # window sizes to be generated in training data
  --stride-kb       20 \
  --min-snps        10 \
  --workers         30 \
  --output          100x_windows.csv
```

### dp-train

Trains a model to predict frequency estimation error.

**Required Files**

- `--snp-freqs-csv` (CSV of SNP frequencies from training populations; columns: chrom, pos, haplotypes, populations)
- `--features-csv` (CSV of window features calculated by `dp-window`)
- `--feature-list` (text file or list specifying which features to use for training)
- `--hap-names-file` (text file listing haplotype names, one per line)

**Example:**

```bash
dp-train \
  --snp-freqs-csv    snp_freqs.csv \ # table of snp frequencies in founders and simulated populations
  --features-csv     features.csv \ # table of features calculuated from simulated populations by dp-window (see walkthrough)
  --feature-list     features.txt \ # list of features to use for training
  --hap-names-file   hap_names.txt \ #list of haplotype names
  --max-snps         400 \ # max SNPs per window (right-cropped)
  --batch            64 \
  --val-batch        128 \
  --epochs           60 \
  --lr               1e-3 \
  --weight-decay     1e-5 \
  --dropout          0.2 \
  --workers          30 \
  --model-name       root_name \
  --save-best
```

### dp-tune

Fine tunes pre-trained models on either specific error ranges or coverage levels.

**Required Files**

- `--snp-freqs-csv` (CSV of SNP frequencies from training populations; same format as `train`)
- `--features-csv` (CSV of window features calculated by `dp-window`)
- `--feature-list` (text file or list of features to use for tuning)
- `--hap-names-file` (text file listing haplotype names, one per line)
- `--pretrained-ckpt` (path to pre-trained checkpoint `.pt` from `haplomatic-train`)

**Example:**

```bash
dp-tune \
  --snp-freqs-csv    100x_freqs.csv \ # snp freq table from 100x simulated populations
  --features-csv     100x_windows.csv \ # features table from dp-window
  --feature-list     features.txt \
  --hap-names-file   hap_names.txt \
  --max-snps         400 \
  --batch            64 \
  --val-batch        128 \
  --epochs           40 \
  --lr               1e-3 \
  --weight-decay     1e-5 \
  --dropout          0.2 \
  --workers          30 \
  --pretrained-ckpt  pre_trained.pt \ # model to be fine-tuned
  --model-name       100x_tuned \ # root name for fine-tuned model
  --save-best
```

### dp-validate

Benchmarks a trained model on held-out data and reports prediction accuracy/verbose logs of genome scans.

**Required Files**

- `--snp-freqs` (CSV of SNP frequencies from training populations; columns: chrom, pos, haplotypes, populations)
- `--features` (CSV of features calculated by `dp-window`)
- `--hap-names` (text file listing haplotype names, one per line)
- `--model` (trained base model from `dp-train`; `.pt` file)
- `--sims` (text file listing simulated populations to evaluate)
- `--regions` (text file listing genomic regions: contig:start:end)
- `--output` (path for validation output CSV or report)
- `--error-threshold` (maximum allowed predicted error for validation)

**Example:**

```bash
dp-validate \
  --snp-freqs        /path/to/freqs.csv \
  --features         features.txt \
  --hap-names-file   hap_names.txt \
  --model            base_model.pt \
  --sims             sim_to_validate.txt \
  --regions          regions.txt \
  --true-freq-dir    /path/to/true_freqs \
  --output           output_name.csv \
  --log-file         log_name.log \
  --error-threshold  .20 \
  --coarse-sizes     30001,50001,60001,70001,80001,90001,100000,150000,200000,250000 \
  --refine-step      5000 \
  --step             20000 \
  --min-snps         30 \
  --max-snps         400 \
  --burnin           100 \
  --sampling         100 \
```

### dp-estimate

Runs adaptive windowing and Bayesian haplotype frequency estimation using a pre-trained model.

**Required Files**

- `--snp-freqs` (CSV with pooled SNP frequencies; columns: chrom, pos, frequencies)
- `--features` (CSV of extracted features for prediction)
- `--hap-names` (text file listing haplotype names, one per line)
- `--model` (trained base model `.pt` from `dp-train`)
- `--sims` (text file listing population or simulation names)
- `--regions` (text file listing genomic regions: contig:start:end)
- `--output` (path for output CSV with predicted error and MCMC frequency estimates)
- `--error-threshold` (maximum allowed predicted error for adaptive windowing)

**Example:**

```bash
dp-estimate \
  --snp-freqs input_freqs.csv \
  --features features.txt \
  --hap-names hap_names.txt \
  --model model.pt \
  --sims sims.txt \
  --regions regions.txt \
  --output estimates.csv \
  --error-threshold 0.2 \
  --coarse-sizes 30001,40001,50001,60001,70001 \
  --refine-step 2500 \
  --step 100000 \
  --min-snps 100 \
  --max-snps 400 \
  --burnin 100 \
  --sampling 100
```

## Documentation

- **Manuscript:** Detailed description of the method, validation, and performance benchmarking. [ADD LINK WHEN READY]

## Citation

If you use DeepPool in your research, please cite:

> Douglas T, Long A, Tarvin R. *Haplomatic: A Deep-Learning Tool for Adaptively Scaling Resolution in Genetic Mapping Studies*. 2025. BioRxiv.

## License

MIT License

Copyright (c) 2025 Tyler E. Douglas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in  
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING  
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS  
IN THE SOFTWARE.

## Contact

For questions, bug reports, or feature requests, please open an issue on GitHub or contact [tyleredouglas](https://github.com/tyleredouglas).
