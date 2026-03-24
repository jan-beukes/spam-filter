# Spam Filter

data is from https://www2.aueb.gr/users/ion/data/enron-spam/

Usage
```sh
rustc spam.rs
./extract-data.sh

./spam # Classify stdin
./spam file # Classify file
./spam test [data_dir] # runs on testing data
```
