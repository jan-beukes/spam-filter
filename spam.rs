use std::collections::HashMap;
use std::io;
use io::{Read, Write};
use std::fs;
use fs::File;
use std::path::Path;

use Classification::*;

const TEST_SET: &str = "data/enron5";
const TRAINING_SET: [&str; 4] = ["data/enron1", "data/enron2", "data/enron3", "data/enron4"];

fn tokenize(content: &str) -> impl Iterator<Item = String> + '_ {
    content.split_whitespace()
        .map(|s| {
            s.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|s| !s.is_empty())
}

// Bag of Words
struct Bow {
    words: HashMap<String, u32>,
    total_documents: u32,
    total_words: u32,
}

impl Bow {
    fn new() -> Bow {
         Bow {
            words: HashMap::<String, u32>::new(),
            total_documents: 0,
            total_words: 0,
        }
    }

    fn from_stdin() -> Result<Bow, io::Error> {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf)?;

        let mut bow = Bow::new();
        bow.load_string(&buf)?;
        Ok(bow)
    }

    fn from_file(path: impl AsRef<Path>) -> Result<Bow, io::Error> {
        let mut bow = Bow::new();
        let content = std::fs::read_to_string(path)?;
        bow.load_string(&content)?;
        Ok(bow)
    }

    fn load_entire_dir(&mut self, dir: &Path) -> Result<(), io::Error> {
        let entries = fs::read_dir(dir)?;
        for entry in entries {
            let path = entry?.path();
            if path.is_dir() {
                continue;
            } else {
                let content = match std::fs::read_to_string(path) {
                    Ok(s) => s,
                    Err(e) => if matches!(e.kind(), io::ErrorKind::InvalidData) {
                        continue;
                    } else {
                        return Err(e);
                    },
                };
                self.load_string(&content)?;
                self.total_documents += 1;
            }
        }
        Ok(())
    }
    
    fn load_string(&mut self, content: &str) -> Result<(), io::Error> {
        for word in tokenize(content) {
            self.total_words += 1;
            self.words.entry(word)
               .and_modify(|freq| { *freq += 1 })
               .or_insert(1);
        }
        Ok(())
    }

}

// type and confidence
enum Classification {
    Spam(f64),
    Ham(f64),
}

impl Classification {
    fn is_spam(&self) -> bool {
        matches!(*self, Classification::Spam(_))
    }

    fn is_ham(&self) -> bool {
        matches!(*self, Classification::Ham(_))
    }
}

struct SpamFilter {
    spam_bow: Bow,
    ham_bow: Bow,
    prob_spam: f64,
    vocab_size: f64,
}

impl SpamFilter {
    fn new() -> SpamFilter {
        SpamFilter {
            spam_bow: Bow::new(),
            ham_bow: Bow::new(),
            prob_spam: 0.0,
            vocab_size: 0.0,
        }
    }

    fn load_from_file(path: &str) -> Result<SpamFilter, io::Error> {
        fn read_bow(string: &str, bow: &mut Bow) {
            let mut lines = string.lines().map(|l| l.trim()).filter(|l| !l.is_empty());
            let header: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
            let [_, docs, words, _] = header[..] else {
                panic!("Could not parse Bow header")
            };
             
            bow.total_documents = docs.parse().unwrap();
            bow.total_words = words.parse().unwrap();
            for line in lines {
                let Some((word, freq)) = line.split_once(":") else {
                    panic!("Could not parse Bow line:\n {}", line);
                };
                bow.words.insert(word.to_string(), freq.parse::<u32>().unwrap());
            }
        }

        let content = fs::read_to_string(path)?;

        let mut sf = SpamFilter::new();
        sf.spam_bow = Bow::new();
        sf.ham_bow = Bow::new();

        let s = &content;
        let end_spam = s.find('}').expect("Expeted '}' in file");
        let end_ham = s.rfind('}').expect("Expeted '}' in file");
        read_bow(&s[..end_spam], &mut sf.spam_bow);
        read_bow(&s[end_spam+1..end_ham], &mut sf.ham_bow);

        let total_documents = sf.spam_bow.total_documents + sf.ham_bow.total_documents;
        sf.prob_spam = sf.spam_bow.total_documents as f64 / total_documents as f64;
        sf.vocab_size = (sf.spam_bow.words.len() + sf.ham_bow.words.len()) as f64;
        Ok(sf)
    }

    fn save_to_file(&self, path: &str) -> Result<(), io::Error> {
        let mut f = File::create(path)?;
        writeln!(&mut f, "spam {} {} {{", self.spam_bow.total_documents, self.spam_bow.total_words)?;
        for word in self.spam_bow.words.keys() {
            writeln!(&mut f, "{word}:{}", self.spam_bow.words[word])?;
        }
        writeln!(&mut f, "}}")?;
        writeln!(&mut f, "ham {} {} {{", self.ham_bow.total_documents, self.ham_bow.total_words)?;
        for word in self.ham_bow.words.keys() {
            writeln!(&mut f, "{word}:{}", self.ham_bow.words[word])?;
        }
        writeln!(&mut f, "}}")?;
        Ok(())
    }

    fn fit_data(&mut self, data_dirs: &[&str]) {
        for dir in data_dirs {
            let dir_path = Path::new(dir);
            self.spam_bow.load_entire_dir(&dir_path.join("spam"))
                .expect("Could not load data from training dir");
            self.ham_bow.load_entire_dir(&dir_path.join("ham"))
                .expect("Could not load data from training dir");
        }

        let total_documents = self.spam_bow.total_documents + self.ham_bow.total_documents;
        self.prob_spam = self.spam_bow.total_documents as f64 / total_documents as f64;
        self.vocab_size = (self.spam_bow.words.len() + self.ham_bow.words.len()) as f64;
    }

    fn predict(&self, bow: &Bow) -> Classification {
        let log_p_spam_given_doc = self.prob_spam.ln() +
            bow.words.keys()
            .fold(0.0, |p, word| {
                let denominator = self.spam_bow.total_words as f64 + self.vocab_size;
                let word_prob = match self.spam_bow.words.get(word) {
                    Some(freq) => (*freq as f64 + 1.0) / denominator,
                    None => 1.0 / denominator,
                };
                p + word_prob.ln()
            });

        let log_p_ham_given_doc = (1.0 - self.prob_spam).ln() +
            bow.words.keys()
            .fold(0.0, |p, word| {
                let denominator = self.ham_bow.total_words as f64 + self.vocab_size;
                let word_prob = match self.ham_bow.words.get(word) {
                    Some(freq) => (*freq as f64 + 1.0) / denominator,
                    None => 1.0 / denominator,
                };
                p + word_prob.ln()
            });

        let diff = log_p_ham_given_doc - log_p_spam_given_doc;
        let confidence = 1.0 / (1.0 + diff.exp());

        if confidence > 0.5 {
            Spam(confidence)
        } else {
            Ham(1.0 - confidence)
        }
    }

    fn test_predictions(&self, data_dir: &str) {
        println!("Testing on {data_dir}");
        let spam_entries = fs::read_dir(Path::new(data_dir).join("spam"))
            .expect("Could not open test spam dir");
        let ham_entries = fs::read_dir(Path::new(data_dir).join("ham"))
            .expect("Could not open test ham dir");

        let mut total = 0;
        let correct_spam = spam_entries.map(|entry| entry.expect("Could not read entry").path())
            .filter(|path| {
                if !path.is_file() { return false }
                let doc = match Bow::from_file(path) {
                    Ok(bow) => bow,
                    Err(e) => if matches!(e.kind(), io::ErrorKind::InvalidData) {
                        return false;
                    } else {
                        panic!("{}", e);
                    },
                };
                total += 1;
                self.predict(&doc).is_spam()
            }).count();

        let correct_ham = ham_entries.map(|entry| entry.expect("Could not read entry").path())
            .filter(|path| {
                if !path.is_file() { return false }
                let doc = match Bow::from_file(path) {
                    Ok(bow) => bow,
                    Err(e) => if matches!(e.kind(), io::ErrorKind::InvalidData) {
                        return false;
                    } else {
                        panic!("{}", e);
                    },
                };
                total += 1;
                self.predict(&doc).is_ham()
            }).count();

        let correct = correct_spam + correct_ham;
        println!("Correctly classified {correct}/{total}");
        println!("Accuracy: {:.4}", correct as f64 / total as f64);
    }
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let mut input_files = None;
    let mut test_dir = None;
    let mut test = false;
    if args.len() > 1 {
        if args[1] == "test" {
            if args.len() > 2 {
                test_dir = Some(&args[2]);
            }
            test = true;
        } else {
            let mut files: Vec<&str> = Vec::new();
            for arg in &args[1..] {
                files.push(arg);
            }
            input_files = Some(files);
        }
    }

    let model_file = "model.sf";
    let clf = match SpamFilter::load_from_file(model_file) {
        Ok(sf) => sf,
        Err(_) => {
            println!("Could not load '{model_file}', retraining...");
            // Train a new one
            let mut sf = SpamFilter::new();
            sf.fit_data(&TRAINING_SET);
            sf.save_to_file(model_file).expect("Could not save file");
            sf
        }
    };

    if test {
        match test_dir {
            Some(dir) => clf.test_predictions(dir),
            None => clf.test_predictions(TEST_SET),
        }
        return
    }

    // Predict document
    let mut results: Vec<(&str, Classification)> = Vec::new();
    match input_files {
        Some(files) => {
            results = files.iter().map(|file| {
                    let doc = Bow::from_file(file).
                        expect("Could not load document");
                    (*file, clf.predict(&doc))
                })
                .collect();
        },
        None => {
            let doc = Bow::from_stdin().expect("Could not load Document from stdin");
            results.push(("stdin", clf.predict(&doc)))
        }
    }

    for result in results {
        match result {
            (file, Spam(confidence)) =>
                println!("{file}: SPAM ({:.4}%)", 100.0*confidence),
            (file, Ham(confidence)) =>
                println!("{file}: NOT SPAM ({:.4}%)", 100.0*confidence),
        }
    }
}
