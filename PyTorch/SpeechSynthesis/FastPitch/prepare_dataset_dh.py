import argparse
import time
import random
from pathlib import Path

import torch
import tqdm
import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
from torch.utils.data import DataLoader

from fastpitch.data_function import TTSCollate, TTSDataset


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-d', '--dataset-path', type=str,
                        default='./', help='Path to dataset')
    parser.add_argument('--wav-text-filelists', required=True, nargs='+',
                        type=str, help='Files with audio paths and text')
    parser.add_argument('--extract-mels', action='store_true',
                        help='Calculate spectrograms from .wav files')
    parser.add_argument('--extract-pitch', action='store_true',
                        help='Extract pitch')
    parser.add_argument('--save-alignment-priors', action='store_true',
                        help='Pre-calculate diagonal matrices of alignment of text to audio')
    parser.add_argument('--log-file', type=str, default='preproc_log.json',
                        help='Filename for logging')
    parser.add_argument('--n-speakers', type=int, default=1)
    # Mel extraction
    parser.add_argument('--max-wav-value', default=32768.0, type=float,
                        help='Maximum audiowave value')
    parser.add_argument('--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--filter-length', default=1024, type=int,
                        help='Filter length')
    parser.add_argument('--hop-length', default=256, type=int,
                        help='Hop (stride) length')
    parser.add_argument('--win-length', default=1024, type=int,
                        help='Window length')
    parser.add_argument('--mel-fmin', default=0.0, type=float,
                        help='Minimum mel frequency')
    parser.add_argument('--mel-fmax', default=8000.0, type=float,
                        help='Maximum mel frequency')
    parser.add_argument('--n-mel-channels', type=int, default=80)
    # Pitch extraction
    parser.add_argument('--f0-method', default='pyin', type=str,
                        choices=['pyin'], help='F0 estimation method')
    # Performance
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('--n-workers', type=int, default=16)
    
    # Language
    parser.add_argument('--symbol_set', default='english_basic',
                        choices=['english_basic', 'english_mandarin_basic'],
                        help='Symbols in the dataset')
    return parser


def read_filelists(filelists):
    """
    Read multiple filelists and return a combined list of (audio_path, text) tuples.
    Assumes each line in the filelist is formatted as: audio_path|text
    """
    entries = []
    for filelist in filelists:
        with open(filelist, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) < 2:
                    continue  # Skip invalid lines
                audio_path, text = parts[0], '|'.join(parts[1:])  # In case text contains '|'
                entries.append((audio_path, text))
    return entries


def split_data(entries, seed, train_ratio=0.75, val_ratio=0.25, test_ratio=0.2):
    """
    Split data into train, val, and test sets based on the provided ratios.
    The train and val splits are derived from the (1 - test_ratio) portion of the data.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    random.shuffle(entries)
    total = len(entries)
    test_size = int(test_ratio * total)
    remaining = total - test_size
    train_size = int(train_ratio * remaining)
    val_size = remaining - train_size
    train_entries = entries[:train_size]
    val_entries = entries[train_size:train_size + val_size]
    test_entries = entries[train_size + val_size:]
    return train_entries, val_entries, test_entries


def write_filelist(entries, filepath, pitch_dir=None):
    """
    Write a list of (audio_path, text) tuples to a filelist.
    If pitch_dir is provided, include pitch_path in the filelist.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for audio_path, text in entries:
            audio_path = Path(audio_path).as_posix()
            if pitch_dir:
                pitch_filename = Path(audio_path).stem + '.pt'
                pitch_path = Path(pitch_dir) / pitch_filename
                pitch_path = pitch_path.as_posix()
                f.write(f"{audio_path}|{pitch_path}|\"{text}\"\n")
            else:
                f.write(f"{audio_path}|\"{text}\"\n")


def preprocess_data(args, all_entries):
    """
    Preprocess all data once: extract mel, pitch, alignment_priors and save to common directories.
    """
    print("\n=== Preprocessing Data ===")

    # Initialize DLLogger
    DLLogger.init(backends=[
        JSONStreamBackend(Verbosity.DEFAULT, Path(args.dataset_path, args.log_file)),
        StdOutBackend(Verbosity.VERBOSE)
    ])
    for k, v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k: v})
    DLLogger.flush()

    # Create directories for preprocessed data
    if args.extract_mels:
        mels_dir = Path(args.dataset_path, 'mels')
        mels_dir.mkdir(parents=True, exist_ok=True)

    if args.extract_pitch:
        pitch_dir = Path(args.dataset_path, 'pitch')
        pitch_dir.mkdir(parents=True, exist_ok=True)

    if args.save_alignment_priors:
        alignment_priors_dir = Path(args.dataset_path, 'alignment_priors')
        alignment_priors_dir.mkdir(parents=True, exist_ok=True)

    # Write a single filelist for preprocessing
    preprocess_filelist = Path(args.dataset_path, 'all_preprocess.txt')
    write_filelist(all_entries, preprocess_filelist)

    # Initialize dataset for preprocessing
    dataset = TTSDataset(
        root_dir=args.dataset_path,
        filelist=preprocess_filelist,
        text_cleaners=['english_cleaners_v2'],
        n_mel_channels=args.n_mel_channels,
        symbol_set=args.symbol_set,
        p_arpabet=0.0,
        n_speakers=args.n_speakers,
        load_mel_from_disk=False,
        load_pitch_from_disk=False,
        pitch_mean=None,
        pitch_std=None,
        max_wav_value=args.max_wav_value,
        sampling_rate=args.sampling_rate,
        filter_length=args.filter_length,
        hop_length=args.hop_length,
        win_length=args.win_length,
        mel_fmin=args.mel_fmin,
        mel_fmax=args.mel_fmax,
        betabinomial_online_dir=None,
        pitch_online_dir=None,
        pitch_online_method=args.f0_method
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=args.n_workers,
        collate_fn=TTSCollate(),
        pin_memory=False,
        drop_last=False
    )

    all_filenames = set()
    for i, batch in enumerate(tqdm.tqdm(data_loader, desc="Preprocessing")):
        tik = time.time()

        _, input_lens, mels, mel_lens, _, pitch, _, _, attn_prior, fpaths = batch

        # Ensure filenames are unique
        for p in fpaths:
            fname = Path(p).name
            if fname in all_filenames:
                raise ValueError(f'Filename is not unique: {fname}')
            all_filenames.add(fname)

        if args.extract_mels:
            for j, mel in enumerate(mels):
                fname = Path(fpaths[j]).with_suffix('.pt').name
                fpath = Path(args.dataset_path, 'mels', fname)
                torch.save(mel[:, :mel_lens[j]], fpath)

        if args.extract_pitch:
            for j, pch in enumerate(pitch):
                fname = Path(fpaths[j]).with_suffix('.pt').name
                fpath = Path(args.dataset_path, 'pitch', fname)
                torch.save(pch[:mel_lens[j]], fpath)

        if args.save_alignment_priors:
            for j, prior in enumerate(attn_prior):
                fname = Path(fpaths[j]).with_suffix('.pt').name
                fpath = Path(args.dataset_path, 'alignment_priors', fname)
                torch.save(prior[:mel_lens[j], :input_lens[j]], fpath)

    print("=== Preprocessing Completed ===\n")
    DLLogger.flush()
    DLLogger.shutdown()


def main():
    parser = argparse.ArgumentParser(description='FastPitch Data Pre-processing with Multiple Random Seeds')
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()
    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    # Read all entries from the provided filelists
    all_entries = read_filelists(args.wav_text_filelists)
    if not all_entries:
        raise ValueError("No valid entries found in the provided filelists.")

    # Preprocess data once
    preprocess_data(args, all_entries)

    # Define 10 random seeds
    seeds = list(range(10))

    for seed in seeds:
        print(f'\n=== Processing Seed {seed} ===')

        # Split the data
        train_entries, val_entries, test_entries = split_data(
            all_entries, seed=seed, test_ratio=0.2, train_ratio=0.75, val_ratio=0.25)

        # Define seed-specific directories
        seed_dir = Path(args.dataset_path) / f'seed_{seed}'
        logs_dir = seed_dir / 'logs'
        seed_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Write split filelists with pitch_path included
        train_filelist = seed_dir / 'train.txt'
        val_filelist = seed_dir / 'val.txt'
        test_filelist = seed_dir / 'test.txt'

        # Assuming pitch files are stored in 'pitch/' directory
        pitch_dir = Path(args.dataset_path, 'pitch')

        write_filelist(train_entries, train_filelist, pitch_dir=pitch_dir)
        write_filelist(val_entries, val_filelist, pitch_dir=pitch_dir)
        write_filelist(test_entries, test_filelist, pitch_dir=pitch_dir)

        # Initialize DLLogger for the current seed
        DLLogger.init(backends=[
            JSONStreamBackend(Verbosity.DEFAULT, logs_dir / f'seed_{seed}_{args.log_file}'),
            StdOutBackend(Verbosity.VERBOSE)
        ])
        DLLogger.log(step="PARAMETER", data={"seed": seed})
        DLLogger.flush()

        # Function to process a given split
        def process_split(split_name, filelist_path):
            print(f'Processing {split_name} split for seed {seed}...')
            dataset = TTSDataset(
                root_dir=args.dataset_path,
                filelist=filelist_path,
                text_cleaners=['english_cleaners_v2'],
                n_mel_channels=args.n_mel_channels,
                symbol_set=args.symbol_set,
                p_arpabet=0.0,
                n_speakers=args.n_speakers,
                load_mel_from_disk=args.extract_mels,
                load_pitch_from_disk=args.extract_pitch,
                pitch_mean=None,
                pitch_std=None,
                max_wav_value=args.max_wav_value,
                sampling_rate=args.sampling_rate,
                filter_length=args.filter_length,
                hop_length=args.hop_length,
                win_length=args.win_length,
                mel_fmin=args.mel_fmin,
                mel_fmax=args.mel_fmax,
                betabinomial_online_dir=None,
                pitch_online_dir=None,
                pitch_online_method=args.f0_method
            )

            data_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=None,
                num_workers=args.n_workers,
                collate_fn=TTSCollate(),
                pin_memory=False,
                drop_last=False
            )

            all_filenames = set()
            for i, batch in enumerate(tqdm.tqdm(data_loader, desc=f"Seed {seed} - {split_name}")):
                tik = time.time()

                # Unpack batch
                # Updated to unpack three fields: audio_path, pitch_path, text
                # Assuming TTSDataset is adjusted to handle the new format
                _, input_lens, mels, mel_lens, _, pitch, _, _, attn_prior, fpaths = batch

                # Ensure filenames are unique
                for p in fpaths:
                    fname = Path(p).name
                    if fname in all_filenames:
                        raise ValueError(f'Filename is not unique: {fname}')
                    all_filenames.add(fname)

                # Since .pt files are already preprocessed and shared across seeds,
                # there's no need to save them again. You can perform any split-specific operations here.
                # For example, you might want to create symbolic links or copy references if needed.

        # Process each split (train, val, test)
        process_split('train', train_filelist)
        process_split('val', val_filelist)
        process_split('test', test_filelist)

        # Flush and close DLLogger for the current seed
        DLLogger.flush()
        DLLogger.shutdown()

    print('\n=== All Seeds Processed Successfully ===')


if __name__ == '__main__':
    main()
