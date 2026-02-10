# dummy class for caching module
class Caching:
    @staticmethod
    def get_dataset_root(dataset, tokenizer, seqlen, cache_base):
        from datasets import load_dataset, VerificationMode
        try:
            from transformers.utils.logging import disable_progress_bar
            disable_progress_bar()
        except:
            pass

        model_name = tokenizer.model_path_or_name.replace("/", "_")

        if dataset == 'c4':
            sys.stdout.write(f"Loading raw c4 dataset...\n")
            raw_datasets = load_dataset('allenai/c4', 'en',
                                        data_files={'train': 'en/c4-train.00000-of-01024.json.gz',
                                                    'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                                        cache_dir=os.path.join(cache_base, 'llm/datasets'),
                                        verification_mode=VerificationMode.NO_CHECKS,
                                        )
            tokenize_function = lambda examples: tokenizer(examples["text"], padding=True,
                                                           truncation=True)
            sys.stdout.write(f"Tokenizing c4 dataset...\n")
            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True,
                                                desc="Running tokenizer on dataset")
            sys.stdout.write(f"Saving tokenized dataset to {os.path.join(cache_base, 'llm/tokenized_datasets', dataset + f"_{model_name}")}...\n")
            os.makedirs(os.path.join(cache_base, 'llm/tokenized_datasets', dataset + f"_{model_name}"), exist_ok=True)
            tokenized_datasets.save_to_disk(os.path.join(cache_base, 'llm/tokenized_datasets', dataset + f"_{model_name}"))
        elif dataset == 'minipile':
            sys.stdout.write(f"Loading raw minipile dataset...\n")
            raw_datasets = load_dataset('JeanKaddour/minipile',
                                        data_files={'train': ['data/train-00000-of-00012-6fbcb5acda05b3c0.parquet',
                                                            'data/train-00001-of-00012-2bb9d088068a84c9.parquet',
                                                            'data/train-00002-of-00012-efb6c8de04272068.parquet',
                                                            'data/train-00003-of-00012-47006e5a888a9324.parquet',
                                                            'data/train-00004-of-00012-a6a94a0207e8e96c.parquet'],
                                                    'validation': 'data/validation-00000-of-00001-a2192e61a091cecb.parquet'},
                                        cache_dir=os.path.join(cache_base, 'llm/datasets'),
                                        verification_mode=VerificationMode.NO_CHECKS,
                                        )
            tokenize_function = lambda examples: tokenizer(examples["text"], padding=True,
                                                           truncation=True)
            sys.stdout.write(f"Tokenizing minipile dataset...\n")
            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True,
                                                desc="Running tokenizer on dataset")
            sys.stdout.write(f"Saving tokenized dataset to {os.path.join(cache_base, 'llm/tokenized_datasets', dataset + f"_{model_name}")}...\n")
            os.makedirs(os.path.join(cache_base, 'llm/tokenized_datasets', dataset + f"_{model_name}"), exist_ok=True)
            tokenized_datasets.save_to_disk(os.path.join(cache_base, 'llm/tokenized_datasets', dataset + f"_{model_name}"))
        elif 'wikitext' in dataset:
            import torch
            sys.stdout.write(f"Loading raw wikitext dataset...\n")
            testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir=os.path.join(cache_base, 'llm/datasets'))
            sys.stdout.write(f"Tokenizing wikitext dataset...\n")
            tokenized_datasets = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
            sys.stdout.write(f"Saving tokenized dataset to {os.path.join(cache_base, 'llm/tokenized_datasets', dataset + f"_{model_name}")}...\n")
            os.makedirs(os.path.join(cache_base, 'llm/tokenized_datasets', dataset + f"_{model_name}"), exist_ok=True)
            torch.save(tokenized_datasets['input_ids'], os.path.join(cache_base, 'llm/tokenized_datasets', dataset + f"_{model_name}", 'input_ids.pt'))
            torch.save(tokenized_datasets['attention_mask'], os.path.join(cache_base, 'llm/tokenized_datasets', dataset + f"_{model_name}", 'attention_mask.pt'))
        else:
            raise ValueError(f"Dataset {dataset} not supported.")

        return os.path.join(cache_base, 'llm/tokenized_datasets', dataset + f"_{model_name}")
