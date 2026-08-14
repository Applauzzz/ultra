# Download Dataset

Here the download_script.sh is the script to download huggingface dataset

This is our way to prepare the dataset with the huggingface form.

```bash
sh download_script.sh  data_cache_path \
                hubcache_path
```

Then, here is a script for jsonl to text file. You need to change your repo_name arg

```bash
sh hf2jsonl.sh  data_cache_path \
                hubcache_path \
                out_dir
```