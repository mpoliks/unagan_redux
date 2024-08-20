Unconditional Audio Generation with GAN and Cycle Regularization
================================================================

This project modifies and expedites the project here:
[Unconditional Audio Generation with GAN and Cycle Regularization](https://arxiv.org/abs/2005.08526)

## Install dependencies
```
pip install -r requirements.txt
```

## Train your own model

Run train.sh, found in /scripts.

## Audio samples

Some generated audio samples can be found in:
```
samples/
```

## Copy over trained melgan files

Note: these steps are from memory there might be steps missing or mistakes.

Assuming the model name `custom` the directory should look like this.

    models/custom/
    ├── mean.mel.npy
    ├── params.generator.hierarchical_with_cycle.pt
    ├── std.mel.npy
    └── vocoder
        ├── args.yml
        ├── modules.py
        ├── params.pt

Assuming `melgan` was trained with

    python scripts/train.py --save_path checkpoints/harmonics --data_path data/harmonics

Assuming the `melgan` directory contains the `melgan` repo, the files to copy are

    mkdir -p models/custom/vocoder
    cp melgan/checkpoints/checkpoint____/files/args.yml models/custom/vocoder/
    cp melgan/mel2wav/modules.py models/custom/vocoder/
    cp checkpoints/checkpoint____/best_netG.pt models/custom/vocoder/params.pt

The `unagan` model also needs to be moved, for example:

    cp training_data/exp_data/{mean,std}.mel.npy models/custom
    cp wandb/checkpoint____/files/save/model/params.Generator.latest.torch models/custom/params.generator.hierarchical_with_cycle.pt

Generation can then be done via

    python generate.py --gid 0 --data_type custom --arch_type hc --duration 10 --num_samples 10

