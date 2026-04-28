"""Run block MCMC sampling on protein sequences using ProGen3 (pre-trained or fine-tuned) model."""

from transformers import LogitsProcessorList

from plm_steer.potentials import load_potentials_from_config
from plm_steer.sampling import LoggingConfig, MCMCConfig, run_block_mcmc_pipeline
from plm_steer.utils.common import get_device, get_default_parser, set_seed
from plm_steer.utils.progen.generate import (
    ProGen3Sampler,
    LengthRangeLogitsProcessor,
    load_progen3_model,
)


def get_args():
    parser = get_default_parser()
    parser.description = (
        "Run block MCMC sampling on protein sequences using ProGen3 (pre-trained or fine-tuned)"
        " model."
    )
    parser.add_argument(
        "--ckpt",
        "-c",
        type=str,
        help="ProGen3 checkpoint path, or HuggingFace Hub pre-trained model name.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=0,
        help="Minimum sequence length (including prompt and special tokens).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length (including prompt and special tokens).",
    )
    args = parser.parse_args()
    if args.save_interval is None:
        args.save_interval = args.num_mcmc_steps
    return args


def main():
    # Get arguments and device
    args = get_args()
    device = get_device()
    if args.seed is not None:
        set_seed(args.seed)

    # Load model and sampler, set MCMC config, define decoding function
    model = load_progen3_model(args.ckpt)

    sampler = ProGen3Sampler(
        model=model,
        beta_sampling=args.beta,
        device=device,
    )
    tokenizer = sampler.batch_preparer.tokenizer

    # In decoding functions, strip tokens '1' and '2', signaling N and C termini
    def decode_batch(ids):
        ids = ids.tolist()
        decoded = tokenizer.decode_batch(ids, skip_special_tokens=True)
        return [s.strip("12") for s in decoded]

    def decode(ids):
        return tokenizer.decode(ids.tolist(), skip_special_tokens=True).strip("12")

    sampling_config = MCMCConfig(
        num_iters=args.num_steps,
        block_size=args.block_size,
        resampling_window=args.resampling_window,
        num_resamples=args.num_mcmc_steps,
    )

    logging_config = LoggingConfig(
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        decode_fn=decode_batch,
        log_interval=10,
        disable_progress_bar=args.disable_progress_bar,
    )

    # define potentials to use
    potentials = None
    if args.potentials_config is not None:
        potentials_transforms = {
            "neighbouring": decode,
            "thermostability": decode,
        }
        potentials = load_potentials_from_config(args.potentials_config, potentials_transforms)

    input_ids = [args.prompt] * args.num_sequences

    # Initialize logits processors
    logits_processors = LogitsProcessorList(
        [
            LengthRangeLogitsProcessor(
                min_length=args.min_length,
                max_length=args.max_length,
                cterm_token_id=sampler.cterm_token_id,
                nterm_token_id=sampler.nterm_token_id,
                pad_token_id=sampler.pad_token_id,
            )
        ]
    )
    generation_kwargs = {"processors": logits_processors}

    # Run block MCMC pipeline
    run_block_mcmc_pipeline(
        input_ids=input_ids,
        sampler=sampler,
        potentials=potentials,
        sampling_config=sampling_config,
        logging_config=logging_config,
        generation_kwargs=generation_kwargs,
    )


if __name__ == "__main__":
    main()
