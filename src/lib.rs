use pyo3::prelude::*;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

const EOT_TOKEN: &str = "<|endoftext|>";

// Serialization format — matches the existing Python JSON schema (string keys).
#[derive(Serialize, Deserialize)]
struct TokenizerData {
    merges: Vec<(u32, u32)>,
    vocab: HashMap<String, Vec<u8>>,
    eot_id: u32,
}

#[pyclass]
pub struct BPETokenizer {
    merges: Vec<(u32, u32)>,
    vocab: HashMap<u32, Vec<u8>>,
    eot_id: u32,
}

#[pymethods]
impl BPETokenizer {
    #[new]
    fn new() -> Self {
        BPETokenizer {
            merges: Vec::new(),
            vocab: HashMap::new(),
            eot_id: 0,
        }
    }

    /// Train BPE on a list of texts.
    ///
    /// `on_merge(step, total, merged_token_bytes)` is called after each merge —
    /// use it to drive a Python-side progress bar.
    #[pyo3(signature = (texts, vocab_size, on_merge=None))]
    fn train(
        &mut self,
        texts: Vec<String>,
        vocab_size: usize,
        on_merge: Option<PyObject>,
        py: Python<'_>,
    ) -> PyResult<()> {
        assert!(vocab_size > 257, "vocab_size must be > 257");
        let n_merges = vocab_size - 256 - 1; // reserve one slot for EOT

        let mut vocab: HashMap<u32, Vec<u8>> =
            (0u32..256).map(|i| (i, vec![i as u8])).collect();

        // Encode corpus as byte-id sequences
        let mut corpus: Vec<Vec<u32>> = texts
            .iter()
            .map(|t| t.as_bytes().iter().map(|&b| b as u32).collect())
            .collect();

        let mut merges: Vec<(u32, u32)> = Vec::with_capacity(n_merges);

        for step in 0..n_merges {
            // Count adjacent pairs
            let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
            for seq in &corpus {
                for w in seq.windows(2) {
                    *counts.entry((w[0], w[1])).or_insert(0) += 1;
                }
            }
            if counts.is_empty() {
                break;
            }

            let best = *counts.iter().max_by_key(|(_, &v)| v).unwrap().0;
            let new_id = 256 + step as u32;
            let merged: Vec<u8> = vocab[&best.0]
                .iter()
                .chain(vocab[&best.1].iter())
                .cloned()
                .collect();
            vocab.insert(new_id, merged.clone());
            merges.push(best);

            // Apply merge across corpus
            for seq in corpus.iter_mut() {
                let mut out = Vec::with_capacity(seq.len());
                let mut i = 0;
                while i < seq.len() {
                    if i + 1 < seq.len() && seq[i] == best.0 && seq[i + 1] == best.1 {
                        out.push(new_id);
                        i += 2;
                    } else {
                        out.push(seq[i]);
                        i += 1;
                    }
                }
                *seq = out;
            }

            // Drive Python progress bar and allow Ctrl-C
            if let Some(ref cb) = on_merge {
                cb.call1(py, (step, n_merges, merged.as_slice()))?;
            }
            py.check_signals()?;
        }

        let eot_id = vocab.len() as u32;
        vocab.insert(eot_id, EOT_TOKEN.as_bytes().to_vec());

        self.merges = merges;
        self.vocab = vocab;
        self.eot_id = eot_id;
        Ok(())
    }

    fn encode(&self, text: &str, add_eot: bool) -> Vec<u32> {
        let mut tokens: Vec<u32> = text.as_bytes().iter().map(|&b| b as u32).collect();

        for (i, &(a, b)) in self.merges.iter().enumerate() {
            let new_id = 256 + i as u32;
            let mut out = Vec::with_capacity(tokens.len());
            let mut j = 0;
            while j < tokens.len() {
                if j + 1 < tokens.len() && tokens[j] == a && tokens[j + 1] == b {
                    out.push(new_id);
                    j += 2;
                } else {
                    out.push(tokens[j]);
                    j += 1;
                }
            }
            tokens = out;
        }

        if add_eot {
            tokens.push(self.eot_id);
        }
        tokens
    }

    fn decode(&self, ids: Vec<u32>) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .filter(|&&id| id != self.eot_id)
            .flat_map(|&id| self.vocab[&id].iter().cloned())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    #[getter]
    fn eot_id(&self) -> u32 {
        self.eot_id
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let data = TokenizerData {
            merges: self.merges.clone(),
            vocab: self
                .vocab
                .iter()
                .map(|(&k, v)| (k.to_string(), v.clone()))
                .collect(),
            eot_id: self.eot_id,
        };
        let json = serde_json::to_string(&data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        std::fs::write(path, json)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(())
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let data: TokenizerData = serde_json::from_str(&json)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(BPETokenizer {
            merges: data.merges,
            vocab: data
                .vocab
                .into_iter()
                .map(|(k, v)| (k.parse::<u32>().unwrap(), v))
                .collect(),
            eot_id: data.eot_id,
        })
    }
}

#[pymodule]
fn _tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BPETokenizer>()?;
    Ok(())
}
