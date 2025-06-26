import os
import tempfile
import pytest
import pandas as pd
import numpy as np
import torch

from deep_pool import (
    estimate,
    pop_simulator,
    read_simulator,
    sim,
    trainer,
    tune,
    validate,
    windows,
)

# ─────────────────────────────────────────────────────────────────────────────
# estimate.py tests
# ─────────────────────────────────────────────────────────────────────────────

def test_read_list_file_success(tmp_path):
    p = tmp_path / "list.txt"
    p.write_text("a\nb\nc\n")
    assert estimate.read_list_file(str(p)) == ["a", "b", "c"]


def test_read_list_file_not_found():
    with pytest.raises(FileNotFoundError):
        estimate.read_list_file("nonexistent.txt")


@pytest.mark.parametrize("X,expected", [
    (np.eye(3), 3.0),
    (np.diag([2,2,0]), 2.0),
])
def test_compute_effective_rank_parametrized(X, expected):
    assert pytest.approx(estimate.compute_effective_rank(X), rel=1e-3) == expected


def test_lsei_haplotype_estimator_shapes_and_errors():
    # valid case
    X = np.eye(2)
    b = np.array([0.5, 0.5])
    p = estimate.lsei_haplotype_estimator(X, b)
    assert pytest.approx(p.sum(), rel=1e-6) == 1.0

    # mismatched dimensions
    X_bad = np.ones((3,2))
    b_bad = np.ones(2)
    with pytest.raises(ValueError):
        estimate.lsei_haplotype_estimator(X_bad, b_bad)


# ─────────────────────────────────────────────────────────────────────────────
# pop_simulator.py tests
# ─────────────────────────────────────────────────────────────────────────────

def test_recombine_shape_and_values(monkeypatch):
    s1 = pd.Series(list("XXXX"))
    s2 = pd.Series(list("YYYY"))
    # force crossover at index 2
    monkeypatch.setattr('random.randrange', lambda a, b: 2)
    c1, c2 = pop_simulator.recombine(s1, s2)
    assert c1.tolist() == ["X", "X", "Y", "Y"]
    assert c2.tolist() == ["Y", "Y", "X", "X"]


def test_get_true_freqs_empty_dataframe():
    df = pd.DataFrame(columns=["hap1", "hap2"])
    freqs = pop_simulator.get_true_freqs(df)
    assert freqs.empty


def test_get_true_freqs_counts():
    df = pd.DataFrame({"hap1": [0,1,1], "hap2": [1,0,1]})
    freqs = pop_simulator.get_true_freqs(df)
    assert set(freqs.columns) == {0,1}
    assert np.allclose(freqs.sum(axis=1), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# read_simulator.py tests
# ─────────────────────────────────────────────────────────────────────────────

def test_read_fasta_static(tmp_path):
    fasta = tmp_path / "test.fa"
    fasta.write_text(">chrA\nACGTACGT\n")
    mapping = read_simulator.ReadSimulator._read_fasta(str(fasta))
    assert mapping == {"chrA": "ACGTACGT"}


def test_reverse_complement_only():
    seq = "ATCG"
    assert read_simulator.ReadSimulator.reverse_complement(seq) == "TAGC"


def test_generate_reads_integration(tmp_path, monkeypatch):
    # create toy FASTA
    fasta = tmp_path / "toy.fa"
    seq = "AAAAACCCCC"
    fasta.write_text(f">hap1\n{seq}\n")
    # setup population DataFrame
    idx = pd.MultiIndex.from_product(
        [["chr1"], list(range(len(seq)))], names=["chrom", "pos"]
    )
    pop = pd.DataFrame({"hap1": ["hap1"]*len(seq)}, index=idx)
    # instantiate simulator
    rs = read_simulator.ReadSimulator({"chr1": str(fasta)}, regions=["chr1"])
    rs._choose_coordinates = lambda population: ("chr1", 0, 3, 3, 3, 6, 3)
    prefix = str(tmp_path / "out")
    hap_counts = rs.generate_reads(pop, n_reads=1, out_prefix=prefix)
    # check FASTQ outputs
    f1 = open(prefix + "_1.fastq").read().splitlines()
    f2 = open(prefix + "_2.fastq").read().splitlines()
    assert len(f1) == 4 and len(f2) == 4
    assert f1[1] == seq[0:3]
    expected_rc = read_simulator.ReadSimulator.reverse_complement(seq[3:6])[::-1]
    assert f2[1] == expected_rc
    assert hap_counts.loc[("chr1", 0), "hap1"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# sim.py tests
# ─────────────────────────────────────────────────────────────────────────────

def test_read_hap_file(tmp_path):
    f = tmp_path / "haps.txt"
    f.write_text("h1\nh2\n")
    assert sim.read_hap_file(str(f)) == ["h1", "h2"]


def test_read_regions_file_malformed(tmp_path):
    f = tmp_path / "regs.txt"
    f.write_text("chr1-no-colon\n")
    with pytest.raises(ValueError):
        sim.read_regions_file(str(f))


def test_read_regions_file_wellformed(tmp_path):
    f = tmp_path / "regs.txt"
    f.write_text("chr2:5-15\n")
    assert sim.read_regions_file(str(f)) == [("chr2",5,15)]


# ─────────────────────────────────────────────────────────────────────────────
# trainer.py tests
# ─────────────────────────────────────────────────────────────────────────────

def test_error_dataset_basic_behavior():
    df_win = pd.DataFrame({"start":[0], "end":[5]})
    snp_df = pd.DataFrame({"pos":[0,5], "hapA":[1,0]})
    idx = np.array([0])
    X_tab = np.random.randn(1,1)
    y = np.array([0.2])
    ds = trainer.ErrorDataset(df_win, snp_df, idx, X_tab, y, max_snps=1, hap_cols=["hapA"])
    raw, tab, label = ds[0]
    assert isinstance(raw, torch.Tensor) and raw.shape == (1,1)
    assert isinstance(tab, torch.Tensor)
    assert isinstance(label, torch.Tensor)


# ─────────────────────────────────────────────────────────────────────────────
# tune.py tests
# ─────────────────────────────────────────────────────────────────────────────

def test_read_list_empty(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("")
    assert tune.read_list(str(p)) == []


def test_read_list_standard(tmp_path):
    p = tmp_path / "input.txt"
    p.write_text("x\ny\nz\n")
    assert tune.read_list(str(p)) == ["x","y","z"]


# ─────────────────────────────────────────────────────────────────────────────
# windows.py tests
# ─────────────────────────────────────────────────────────────────────────────

def test_windows_effective_rank_edge():
    # For a zero matrix, all singular values are zero => entropy=0 => exp(entropy)=1
    X = np.zeros((3,3))
    assert pytest.approx(estimate.compute_effective_rank(X), rel=1e-6) == 1.0
    assert pytest.approx(windows.compute_effective_rank(X), rel=1e-6) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# validate.py tests
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_avg_snpfreqs_empty():
    df = pd.DataFrame({"B1":[0,0],"B2":[0,0],"snp":[1,2]})
    out = validate.compute_avg_snpfreqs(df, sim="snp")
    assert np.all(np.isnan(out))


def test_compute_avg_snpfreqs_basic():
    df = pd.DataFrame({"B1":[1,0],"B2":[0,1],"snp":[1.0,3.0]})
    out = validate.compute_avg_snpfreqs(df, sim="snp")
    assert pytest.approx(out[0]) == 1.0
    assert pytest.approx(out[1]) == 3.0


def test_validate_effective_rank():
    X = np.eye(4)
    assert pytest.approx(validate.compute_effective_rank(X), rel=1e-3) == 4.0
