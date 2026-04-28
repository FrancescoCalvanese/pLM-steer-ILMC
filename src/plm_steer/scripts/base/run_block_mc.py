"""Run MCMC steering on pre-trained GPT-like simple transformer model for protein sequences."""

from plm_steer.potentials import load_potentials_from_config
from plm_steer.sampling import LoggingConfig, MCMCConfig, run_block_mcmc_pipeline
from plm_steer.utils.common import get_device, get_default_parser, set_seed
from plm_steer.model.sampler import GPTSampler
from plm_steer.model.tokenizer import GPTTokenizer
from plm_steer.model.transformer import GPTTransformer, load_model


def get_args():
    parser = get_default_parser()
    parser.description = (
        "Run block MCMC sampling on protein sequences using a pre-trained GPT-style model."
    )
    parser.add_argument(
        "--ckpt",
        "-c",
        type=str,
        help="Pretrained checkpoint path. The same directory should contain a JSON config file.",
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
    tokenizer = GPTTokenizer()  # default
    model: GPTTransformer = load_model(args.ckpt)
    model = model.eval().to(device)

    sampling_config = MCMCConfig(
        num_iters=args.num_steps,
        block_size=args.block_size,
        resampling_window=args.resampling_window,
        num_resamples=args.num_mcmc_steps,
    )

    sampler = GPTSampler(
        model=model,
        beta_sampling=args.beta,
        device=device,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    def decode_batch(ids):
        return [tokenizer.decode(id_seq, skip_special_tokens=True) for id_seq in ids]

    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=True)

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

    input_ids = [tokenizer.bos_token + args.prompt] * args.num_sequences
    input_ids = tokenizer(
        input_ids,
        add_special_tokens=False,
        return_tensors="pt",
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"].to(device)

    # Run block MCMC pipeline
    run_block_mcmc_pipeline(
        input_ids=input_ids,
        sampler=sampler,
        potentials=potentials,
        sampling_config=sampling_config,
        logging_config=logging_config,
    )


if __name__ == "__main__":
    main()
